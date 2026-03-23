# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""RMSNorm kernel builder using the @flyc.kernel API.

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

Entry points:

- ``build_rmsnorm_module(M, N, dtype, with_residual=False)`` — launch
  ``(Input, Gamma, Output, m, stream)``.
- ``build_rmsnorm_module(..., with_residual=True)`` or
  ``build_rmsnorm_module_with_residual`` — fuses ``x = Input + Residual``, writes ``x``
  **in place** into ``Residual``, then ``Output = RMSNorm(x) * gamma``;
  launch ``(Input, Residual, Gamma, Output, m, stream)``.

Two compute paths:

- Fast path (N % tile_cols == 0): buffer_load/store vectorised access.
- Generic path (arbitrary N): scalar copy_atom_call.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import arith, vector, gpu, range_constexpr
from flydsl.expr.typing import T, Int32
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from flydsl._mlir import ir
from flydsl.expr import buffer_ops


KERNEL_NAME = "rmsnorm"

EPS = 1e-5

BLOCK_THREADS = 256
WARP_SIZE = 64
VEC_WIDTH = 8


def dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f32":
        return T.f32
    if dtype_str == "f16":
        return T.f16
    if dtype_str == "bf16":
        return T.bf16
    raise ValueError(f"unsupported dtype: {dtype_str}")



def build_rmsnorm_module(M: int, N: int, dtype_str: str, with_residual: bool = False):
    if with_residual:
        return _build_rmsnorm_with_residual(M, N, dtype_str)
    return _build_rmsnorm_no_residual(M, N, dtype_str)


def build_rmsnorm_module_with_residual(M: int, N: int, dtype_str: str):
    """Same as :func:`build_rmsnorm_module` with ``with_residual=True``."""
    return build_rmsnorm_module(M, N, dtype_str, with_residual=True)


def _build_rmsnorm_no_residual(M: int, N: int, dtype_str: str):
    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    tile_cols = BLOCK_THREADS * VEC_WIDTH # 2048
    # 这个地方是4 几个wae的意思
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16

    allocator = SmemAllocator(None, arch=arch)
    f32_bytes = 4
    red_offset = allocator._align(allocator.ptr, 16) # 从当前游标对齐到 16，作为第一块起点
    allocator.ptr = red_offset + RED_SLOTS * f32_bytes #  游标移到第一块结束之后
    red2_offset = allocator._align(allocator.ptr, 16) # 从「第一块之后」再对齐，作为第二块
    allocator.ptr = red2_offset + RED_SLOTS * f32_bytes
    """
    // 同一块 unsigned char smem[TOTAL];
    // float *s_red  = (float*)(smem + red_offset);
    // float *s_red2 = (float*)(smem + red2_offset);
    """
    @flyc.kernel
    def rmsnorm_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32

        fm_fast = arith.FastMathFlags.fast
        eps_c = arith.constant(EPS, type=compute_type)
        n_float = arith.constant(float(N), type=compute_type)

        base_ptr = allocator.get_base()
        s_red = SmemPtr(base_ptr, red_offset, T.f32, shape=(RED_SLOTS,))
        s_red2 = SmemPtr(base_ptr, red2_offset, T.f32, shape=(RED_SLOTS,))
        s_red.get()
        s_red2.get()
        """
        SmemPtr.get() 在实现里是 memref.view：在 同一个 base_memref 上，按 byte_offset 切出一个带类型的 memref 视图，方便后面的 load/store：
        smem_allocator.py
        Lines 88-108
            def get(self) -> ir.Value:

                ...
                offset_op = arith.constant(T.index(), self.byte_offset)
                ...
                self._view_cache = memref.view(target_type, self.base_memref, offset_val, sizes=[])
                return self._view_cache
        """

        def wave_reduce_add(x):
            width_i32 = fx.Int32(WARP_SIZE)
            w = x
            for sh in [32, 16, 8, 4, 2, 1]:
                off = fx.Int32(sh)
                peer = w.shuffle_xor(off, width_i32)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            if RED_SLOTS == 1:
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == fx.Int32(0):
                wave_idx = arith.index_cast(T.index, wave)
                s_red.store(w0, [wave_idx])
                s_red2.store(w1, [wave_idx])
            gpu.barrier()

            if wave == fx.Int32(0):
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, fx.Int32(0))
                lane_safe_idx = arith.index_cast(T.index, lane_safe)
                v0 = s_red.load([lane_safe_idx])
                v1 = s_red2.load([lane_safe_idx])
                z = fx.Float32(0.0)
                ww0 = in_range.select(v0, z)
                ww1 = in_range.select(v1, z)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == fx.Int32(0):
                    c0_idx = fx.Index(0)
                    s_red.store(ww0, [c0_idx])
                    s_red2.store(ww1, [c0_idx])
            gpu.barrier()

            c0_idx = fx.Index(0)
            return s_red.load([c0_idx]), s_red2.load([c0_idx])

        # ==================================================================
        # Fast path: N is a multiple of tile_cols
        # ==================================================================
        if N >= tile_cols and N % tile_cols == 0 and elem_bits <= 16:
            from flydsl.expr.arith import ArithValue

            num_tiles = N // tile_cols
            elem_bytes = 4 if dtype_str == "f32" else 2
            vec_dwords = (VEC_WIDTH * elem_bytes) // 4

            vec_type_c = T.vec(VEC_WIDTH, compute_type)
            vec_type_e = T.vec(VEC_WIDTH, elem_type)

            in_rsrc = buffer_ops.create_buffer_resource(Input, max_size=True)
            out_rsrc = buffer_ops.create_buffer_resource(Output, max_size=True)
            gamma_rsrc = buffer_ops.create_buffer_resource(Gamma, max_size=True)

            # 当前行在扁平 buffer 里的字节起点：第 bid 行 × 每行 N 个元素 × 每元素字节数。
            # bid 包成 ArithValue 才能和后面的标量/向量表达式一起做 MLIR 算术（与 thr_col_bytes 等同理）。
            # 这里有个小发现 这个偏移按byte算的
            row_soffset = ArithValue(bid) * (N * elem_bytes)
            thr_col_bytes = ArithValue(tid) * (VEC_WIDTH * elem_bytes)

            def _load_vec(rsrc, col_byte_off, soff=None):
                # col_byte_off 按字节；buffer_load(..., dtype=i32) 的 offset 按 dword（4 字节）计，故 /4 即 >>2。
                dw = col_byte_off >> fx.Int32(2)
                # soffset_bytes=soff = 把 soff（这里是行起点字节偏移）作为 buffer 指令的标量字节偏移；soff=None 时表示不加这段行偏移。
                raw = buffer_ops.buffer_load(rsrc, dw, vec_width=vec_dwords, dtype=T.i32, soffset_bytes=soff)
                """
                都是 i32 打包 load → 视作 8 宽元素向量；
                f32 时 dword 数等于 VEC_WIDTH，用 raw.bitcast；bf16/f16 时 dword 更少，
                用 vector.bitcast 把 vec<4xi32> 转成 vec<8xbf16>（或 f16）。
                """
                if vec_dwords == VEC_WIDTH:
                    return raw.bitcast(vec_type_e)
                return vector.bitcast(vec_type_e, raw)

            def _store_vec(data, rsrc, col_byte_off, soff=None):
                # 与 _load_vec 相同：列方向字节偏移 → dword 下标。
                dw = col_byte_off >> fx.Int32(2)
                buffer_ops.buffer_store(data, rsrc, dw, soffset_bytes=soff)

            c_zero_f = arith.constant(0.0, type=compute_type)
            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            cache_as_elem = (dtype_str != "f32")
            # # in_local 是 编译期收集 SSA 向量用的 Python 列表；生成代码里对应 
            # 每线程一份「向量寄存器里的缓存」（理想情况），存的是 该线程在行上的分片，不是整行。
            in_local = [] 

            # Pass 1: load + cache + sumsq
            for tile_i in range_constexpr(num_tiles):
                col_bytes = ArithValue(thr_col_bytes) + (tile_i * tile_cols * elem_bytes)
                vec_e = _load_vec(in_rsrc, col_bytes, soff=row_soffset)

                if cache_as_elem:
                    in_local.append(vec_e)
                    x = vec_e.extf(vec_type_c) #看上去像 arith.extpf https://mlir.llvm.org/docs/Dialects/ArithOps/#arithextf-arithextfop
                else: #fp32 分支不考虑
                    x = vec_e
                    in_local.append(x)

                x_av = ArithValue(x)
                # TODO 有没有一把ADD的 指令 308上汇编是八条了
                x2 = x_av * x_av
                red2 = vector.reduction(compute_type, vector.CombiningKind.ADD, x2, fastmath=fm_fast)
                thread_sumsq = ArithValue(thread_sumsq) + red2

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = ArithValue(sum_sq) / n_float
            ms_eps = ArithValue(mean_sq) + eps_c
            rrms = ms_eps.rsqrt(fastmath=fm_fast)
            # 先把 行内共用的标量尺度 rrms 扩成 8 lane 向量，再包一层 ArithValue，以便和 8×f32 的 x 做向量乘法。
            rrms_splat = vector.broadcast(vec_type_c, rrms)
            rrms_splat_av = ArithValue(rrms_splat)

            # Pass 2: normalize + gamma + store (reuse cached input)
            for tile_i in range_constexpr(num_tiles):
                col_bytes = ArithValue(thr_col_bytes) + (tile_i * tile_cols * elem_bytes)

                g_e = _load_vec(gamma_rsrc, col_bytes) # weight
                g = g_e if dtype_str == "f32" else g_e.extf(vec_type_c)

                x = in_local[tile_i]
                if cache_as_elem:
                    x = x.extf(vec_type_c)

                x_av = ArithValue(x)
                g_av = ArithValue(g)
                y = (x_av * rrms_splat_av) * g_av
                y_val = y

                if dtype_str == "bf16":
                    if USE_HW_CVT_PK_BF16_F32:
                        out_e = y_val.truncf(vec_type_e)
                    else:
                        vec_i32_ty = T.vec(VEC_WIDTH, T.i32)
                        vec4_i32_ty = T.vec(VEC_WIDTH // 2, T.i32)
                        vec_bf16_ty = T.vec(VEC_WIDTH, elem_type)
                        c16_i32 = arith.constant(16, type=T.i32)
                        c16_v = vector.broadcast(vec_i32_ty, c16_i32)
                        u = y_val.bitcast(vec_i32_ty)
                        upper = u.shrui(c16_v)
                        c1_v = vector.broadcast(vec_i32_ty, arith.constant(1, type=T.i32))
                        lsb = upper & c1_v
                        c7fff_v = vector.broadcast(vec_i32_ty, arith.constant(0x7FFF, type=T.i32))
                        bias = ArithValue(c7fff_v) + ArithValue(lsb)
                        u_round = ArithValue(u) + bias
                        bf16_bits = u_round.shrui(c16_v)
                        even = vector.shuffle(bf16_bits, bf16_bits, [0, 2, 4, 6])
                        odd = vector.shuffle(bf16_bits, bf16_bits, [1, 3, 5, 7])
                        odd_sh = odd << vector.broadcast(vec4_i32_ty, c16_i32)
                        packed = even | odd_sh
                        out_e = vector.bitcast(vec_bf16_ty, packed)
                elif dtype_str == "f32":
                    out_e = y_val
                else:
                    out_e = y_val.truncf(vec_type_e)

                out_col = ArithValue(thr_col_bytes) + (tile_i * tile_cols * elem_bytes)
                i32_vec_ty = T.vec(vec_dwords, T.i32)
                # 把 out_e 按位 看成 vec<vec_dwords × i32> 再 store；f32 时 8 lane→8 i32 用 out_e.bitcast，
                # bf16 时 8×bf16→4×i32 用 vector.bitcast，因为 向量形状在 MLIR 里不一样
                out_vec = vector.bitcast(i32_vec_ty, out_e) if vec_dwords != VEC_WIDTH else out_e.bitcast(i32_vec_ty)
                _store_vec(out_vec, out_rsrc, out_col, soff=row_soffset)

        else:
            # ==============================================================
            # Generic path: scalar 2-pass for arbitrary N
            # ==============================================================
            from flydsl.expr.arith import ArithValue

            row_in = fx.slice(Input, (bid, None))
            row_out = fx.slice(Output, (bid, None))

            copy_atom_s = fx.make_copy_atom(fx.UniversalCopy(elem_bits), elem_bits)
            scalar_reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
            scalar_reg_lay = fx.make_layout(1, 1)

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            def _load_scalar(divided_tensor, index):
                view = fx.slice(divided_tensor, (None, index))
                r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
                fx.copy_atom_call(copy_atom_s, view, r)
                v = fx.memref_load_vec(r)
                return vector.extract(v, static_position=[0])

            def _store_scalar(divided_tensor, index, val):
                r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
                vec_ty = T.vec(1, elem_type)
                v = vector.from_elements(vec_ty, [val])
                fx.memref_store_vec(v, r)
                view = fx.slice(divided_tensor, (None, index))
                fx.copy_atom_call(copy_atom_s, r, view)

            c_zero_f = arith.constant(0.0, type=compute_type)
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                c_N_i32 = Int32(N)
                is_valid = idx < c_N_i32
                c0_i = Int32(0)
                idx_safe = is_valid.select(idx, c0_i)
                x_e = _load_scalar(row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.extf(compute_type)
                x_av = ArithValue(x)
                x2 = x_av * x_av
                x2_safe = is_valid.select(x2, c_zero_f)
                thread_sumsq = ArithValue(thread_sumsq) + x2_safe

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = ArithValue(sum_sq) / n_float
            ms_eps = ArithValue(mean_sq) + eps_c
            rrms = ms_eps.rsqrt(fastmath=fm_fast)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                c_N_i32 = Int32(N)
                if arith.cmpi(arith.CmpIPredicate.ult, idx, c_N_i32):
                    x_e = _load_scalar(row_div, idx)
                    g_e = _load_scalar(gamma_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.extf(compute_type)
                    g = g_e if dtype_str == "f32" else g_e.extf(compute_type)
                    norm = ArithValue(x) * ArithValue(rrms)
                    y = norm * ArithValue(g)
                    if dtype_str == "f32":
                        y_e = y
                    elif dtype_str == "bf16":
                        y_e = y.truncf(elem_type)
                    else:
                        y_e = y.truncf(elem_type)
                    _store_scalar(out_div, idx, y_e)

    @flyc.jit
    def launch_rmsnorm(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        idx_m = arith.index_cast(T.index, m_in) # 把 m_in 从 int32 转成 index
        launcher = rmsnorm_kernel(Input, Gamma, Output)
        launcher.launch(
            grid=(idx_m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm


def _build_rmsnorm_with_residual(M: int, N: int, dtype_str: str):
    """RMSNorm on (Input + Residual); fused sum in place to Residual; norm*gamma to Output."""
    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    tile_cols = BLOCK_THREADS * VEC_WIDTH
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16

    allocator = SmemAllocator(None, arch=arch)
    f32_bytes = 4
    red_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = red_offset + RED_SLOTS * f32_bytes
    red2_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = red2_offset + RED_SLOTS * f32_bytes

    @flyc.kernel
    def rmsnorm_kernel_residual(
        Input: fx.Tensor,
        Residual: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32

        fm_fast = arith.FastMathFlags.fast
        eps_c = arith.constant(EPS, type=compute_type)
        n_float = arith.constant(float(N), type=compute_type)

        base_ptr = allocator.get_base()
        s_red = SmemPtr(base_ptr, red_offset, T.f32, shape=(RED_SLOTS,))
        s_red2 = SmemPtr(base_ptr, red2_offset, T.f32, shape=(RED_SLOTS,))
        s_red.get()
        s_red2.get()

        def wave_reduce_add(x):
            width_i32 = fx.Int32(WARP_SIZE)
            w = x
            for sh in [32, 16, 8, 4, 2, 1]:
                off = fx.Int32(sh)
                peer = w.shuffle_xor(off, width_i32)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            if RED_SLOTS == 1:
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == fx.Int32(0):
                wave_idx = arith.index_cast(T.index, wave)
                s_red.store(w0, [wave_idx])
                s_red2.store(w1, [wave_idx])
            gpu.barrier()

            if wave == fx.Int32(0):
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, fx.Int32(0))
                lane_safe_idx = arith.index_cast(T.index, lane_safe)
                v0 = s_red.load([lane_safe_idx])
                v1 = s_red2.load([lane_safe_idx])
                z = fx.Float32(0.0)
                ww0 = in_range.select(v0, z)
                ww1 = in_range.select(v1, z)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == fx.Int32(0):
                    c0_idx = fx.Index(0)
                    s_red.store(ww0, [c0_idx])
                    s_red2.store(ww1, [c0_idx])
            gpu.barrier()

            c0_idx = fx.Index(0)
            return s_red.load([c0_idx]), s_red2.load([c0_idx])

        if N >= tile_cols and N % tile_cols == 0 and elem_bits <= 16:
            from flydsl.expr.arith import ArithValue

            num_tiles = N // tile_cols
            elem_bytes = 4 if dtype_str == "f32" else 2
            vec_dwords = (VEC_WIDTH * elem_bytes) // 4

            vec_type_c = T.vec(VEC_WIDTH, compute_type)
            vec_type_e = T.vec(VEC_WIDTH, elem_type)

            in_rsrc = buffer_ops.create_buffer_resource(Input, max_size=True)
            res_rsrc = buffer_ops.create_buffer_resource(Residual, max_size=True)
            out_rsrc = buffer_ops.create_buffer_resource(Output, max_size=True)
            gamma_rsrc = buffer_ops.create_buffer_resource(Gamma, max_size=True)

            row_soffset = ArithValue(bid) * (N * elem_bytes)
            thr_col_bytes = ArithValue(tid) * (VEC_WIDTH * elem_bytes)

            def _load_vec(rsrc, col_byte_off, soff=None):
                dw = col_byte_off >> fx.Int32(2)
                raw = buffer_ops.buffer_load(rsrc, dw, vec_width=vec_dwords, dtype=T.i32, soffset_bytes=soff)
                if vec_dwords == VEC_WIDTH:
                    return raw.bitcast(vec_type_e)
                return vector.bitcast(vec_type_e, raw)

            def _store_vec(data, rsrc, col_byte_off, soff=None):
                dw = col_byte_off >> fx.Int32(2)
                buffer_ops.buffer_store(data, rsrc, dw, soffset_bytes=soff)

            c_zero_f = arith.constant(0.0, type=compute_type)
            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            fused_local = []

            for tile_i in range_constexpr(num_tiles):
                col_bytes = ArithValue(thr_col_bytes) + (tile_i * tile_cols * elem_bytes)
                vec_in = _load_vec(in_rsrc, col_bytes, soff=row_soffset)
                vec_r = _load_vec(res_rsrc, col_bytes, soff=row_soffset)
                x = ArithValue(vec_in.extf(vec_type_c)) + ArithValue(vec_r.extf(vec_type_c))
                fused_local.append(x)
                x_av = ArithValue(x)
                x2 = x_av * x_av
                red2 = vector.reduction(compute_type, vector.CombiningKind.ADD, x2, fastmath=fm_fast)
                thread_sumsq = ArithValue(thread_sumsq) + red2

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = ArithValue(sum_sq) / n_float
            ms_eps = ArithValue(mean_sq) + eps_c
            rrms = ms_eps.rsqrt(fastmath=fm_fast)
            rrms_splat = vector.broadcast(vec_type_c, rrms)
            rrms_splat_av = ArithValue(rrms_splat)

            if dtype_str == "bf16" and not USE_HW_CVT_PK_BF16_F32:
                vec_i32_ty_bf = T.vec(VEC_WIDTH, T.i32)
                vec4_i32_ty_bf = T.vec(VEC_WIDTH // 2, T.i32)
                vec_bf16_ty_bf = T.vec(VEC_WIDTH, elem_type)
                c16_i32_bf = arith.constant(16, type=T.i32)
                c16_v_bf = vector.broadcast(vec_i32_ty_bf, c16_i32_bf)

                def _bf16_round_pack(val):
                    u = val.bitcast(vec_i32_ty_bf)
                    upper = u.shrui(c16_v_bf)
                    c1_v = vector.broadcast(vec_i32_ty_bf, arith.constant(1, type=T.i32))
                    lsb = upper & c1_v
                    c7fff_v = vector.broadcast(vec_i32_ty_bf, arith.constant(0x7FFF, type=T.i32))
                    bias = ArithValue(c7fff_v) + ArithValue(lsb)
                    u_round = ArithValue(u) + bias
                    bf16_bits = u_round.shrui(c16_v_bf)
                    even = vector.shuffle(bf16_bits, bf16_bits, [0, 2, 4, 6])
                    odd = vector.shuffle(bf16_bits, bf16_bits, [1, 3, 5, 7])
                    odd_sh = odd << vector.broadcast(vec4_i32_ty_bf, c16_i32_bf)
                    packed = even | odd_sh
                    return vector.bitcast(vec_bf16_ty_bf, packed)
            else:
                _bf16_round_pack = None  # unused

            for tile_i in range_constexpr(num_tiles):
                col_bytes = ArithValue(thr_col_bytes) + (tile_i * tile_cols * elem_bytes)

                g_e = _load_vec(gamma_rsrc, col_bytes)
                g = g_e if dtype_str == "f32" else g_e.extf(vec_type_c)

                x_fused = fused_local[tile_i]
                x_av = ArithValue(x_fused)
                g_av = ArithValue(g)
                y = (x_av * rrms_splat_av) * g_av
                y_val = y

                if dtype_str == "bf16":
                    if USE_HW_CVT_PK_BF16_F32:
                        out_e = y_val.truncf(vec_type_e)
                        x_res_e = x_fused.truncf(vec_type_e)
                    else:
                        out_e = _bf16_round_pack(y_val)
                        x_res_e = _bf16_round_pack(x_fused)
                elif dtype_str == "f32":
                    out_e = y_val
                    x_res_e = x_fused
                else:
                    out_e = y_val.truncf(vec_type_e)
                    x_res_e = x_fused.truncf(vec_type_e)

                out_col = ArithValue(thr_col_bytes) + (tile_i * tile_cols * elem_bytes)
                i32_vec_ty = T.vec(vec_dwords, T.i32)
                out_vec = vector.bitcast(i32_vec_ty, out_e) if vec_dwords != VEC_WIDTH else out_e.bitcast(i32_vec_ty)
                res_inpl_vec = vector.bitcast(i32_vec_ty, x_res_e) if vec_dwords != VEC_WIDTH else x_res_e.bitcast(i32_vec_ty)
                _store_vec(out_vec, out_rsrc, out_col, soff=row_soffset)
                _store_vec(res_inpl_vec, res_rsrc, out_col, soff=row_soffset)

        else:
            from flydsl.expr.arith import ArithValue

            row_in = fx.slice(Input, (bid, None))
            row_res = fx.slice(Residual, (bid, None))
            row_out = fx.slice(Output, (bid, None))

            copy_atom_s = fx.make_copy_atom(fx.UniversalCopy(elem_bits), elem_bits)
            scalar_reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
            scalar_reg_lay = fx.make_layout(1, 1)

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            row_res_div = fx.logical_divide(row_res, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            def _load_scalar(divided_tensor, index):
                view = fx.slice(divided_tensor, (None, index))
                r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
                fx.copy_atom_call(copy_atom_s, view, r)
                v = fx.memref_load_vec(r)
                return vector.extract(v, static_position=[0])

            def _store_scalar(divided_tensor, index, val):
                r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
                vec_ty = T.vec(1, elem_type)
                v = vector.from_elements(vec_ty, [val])
                fx.memref_store_vec(v, r)
                view = fx.slice(divided_tensor, (None, index))
                fx.copy_atom_call(copy_atom_s, r, view)

            c_zero_f = arith.constant(0.0, type=compute_type)
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                c_N_i32 = Int32(N)
                is_valid = idx < c_N_i32
                c0_i = Int32(0)
                idx_safe = is_valid.select(idx, c0_i)
                in_e = _load_scalar(row_div, idx_safe)
                r_e = _load_scalar(row_res_div, idx_safe)
                x_in = in_e if dtype_str == "f32" else in_e.extf(compute_type)
                x_r = r_e if dtype_str == "f32" else r_e.extf(compute_type)
                x = ArithValue(x_in) + ArithValue(x_r)
                x2 = x * x
                x2_safe = is_valid.select(x2, c_zero_f)
                thread_sumsq = ArithValue(thread_sumsq) + x2_safe

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = ArithValue(sum_sq) / n_float
            ms_eps = ArithValue(mean_sq) + eps_c
            rrms = ms_eps.rsqrt(fastmath=fm_fast)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                c_N_i32 = Int32(N)
                if arith.cmpi(arith.CmpIPredicate.ult, idx, c_N_i32):
                    in_e = _load_scalar(row_div, idx)
                    r_e = _load_scalar(row_res_div, idx)
                    x = in_e if dtype_str == "f32" else in_e.extf(compute_type)
                    x = ArithValue(x) + ArithValue(r_e if dtype_str == "f32" else r_e.extf(compute_type))
                    g_e = _load_scalar(gamma_div, idx)
                    g = g_e if dtype_str == "f32" else g_e.extf(compute_type)
                    norm = ArithValue(x) * ArithValue(rrms)
                    y = norm * ArithValue(g)
                    if dtype_str == "f32":
                        y_e = y
                        x_sum_e = x
                    elif dtype_str == "bf16":
                        y_e = y.truncf(elem_type)
                        x_sum_e = x.truncf(elem_type)
                    else:
                        y_e = y.truncf(elem_type)
                        x_sum_e = x.truncf(elem_type)
                    _store_scalar(out_div, idx, y_e)
                    _store_scalar(row_res_div, idx, x_sum_e)

    @flyc.jit
    def launch_rmsnorm_residual(
        Input: fx.Tensor,
        Residual: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        idx_m = arith.index_cast(T.index, m_in)
        launcher = rmsnorm_kernel_residual(Input, Residual, Gamma, Output)
        launcher.launch(
            grid=(idx_m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_residual
