# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

try:
    import torch
except ImportError:
    torch = None

import cuquantum


# note: this implementation would cause pytorch tests being silently skipped
# if pytorch is not available, which is the desired effect since otherwise
# it'd be too noisy
backend_names = ("numpy", "cupy")
if torch:
    backend_names += ("torch-cpu", "torch-gpu")


dtype_names = (
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
)


# the expressions here should be
#   - a plain einsum expression (subscript, or interleaved as a tuple)
#     - for interleaved, the output modes can be explicitly given or left as None
#   - a list [einsum_expr, network_options, optimizer_options, overwrite_dtype]
# the second variant is suitable for testing exotic TNs that require further customization
# TODO: expand the tests
einsum_expressions = (
    "ea,fb,abcd,gc,hd->efgh",
    "ea,fb,abcd,gc,hd",
    "ij,jk,kl->il",
    "ij,jk,kl",
    "ij,jk,ki",
    "abc,bcd->",
    "ab,bc,ca->",
    "abc,ace,abd->de",
    "abc,ace,abd->ade",
    ["...ik,...k,...kj->...ij", {}, {}, "complex128"],  # SVD reconstruction
    ((2, 3, 4), (3, 4, 5), (2, 1), (1, 5), None),
    (('a', 'b'), ('b', 'c', 'd'), ('a',)),  # opt_einsum and cutensornet support this, but not NumPy et al
    [((5, 4, 3), (3, 4, 6), (6, 5), None), {}, {}, "float32"],
    ["abc,bcd,ade", {}, {"slicing": {"min_slices": 4}}, "float64"],
    # TODO: need >8 operand tests (from L0)
    ["ao,bp,cq,dr,es,ft,gu,hv,iw,jx,ky,lz,mA,nB,oP,pO,qN,rM,sL,tK,uJ,vI,wH,xG,yF,zE,AD,BC,DC,FC,PC,CQ,FD,ID,DR,JE,KE,PE,ES,GF,FT,LG,NG,GU,IH,JH,MH,HV,KI,IW,KJ,JX,KY,NL,OL,LZ,NM,OM,MÀ,NÁ,PO,OÂ,PÃ,RQ,TQ,ÃQ,QÄ,TR,WR,RÅ,XS,YS,ÃS,SÆ,UT,TÇ,ZU,ÁU,UÈ,WV,XV,ÀV,VÉ,YW,WÊ,YX,XË,YÌ,ÁZ,ÂZ,ZÍ,ÁÀ,ÂÀ,ÀÎ,ÁÏ,ÃÂ,ÂÐ,ÃÑ,Äß,ÅÞ,ÆÝ,ÇÜ,ÈÛ,ÉÚ,ÊÙ,ËØ,Ì×,ÍÖ,ÎÕ,ÏÔ,ÐÓ,ÑÒ->", {}, {}, "float64"],  # QAOA MaxCut
    ["ab,bc->ac", {'compute_type': cuquantum.ComputeType.COMPUTE_64F}, {}, "complex128"],

    # CuPy large TN tests
    ["a,b,c->abc", {}, {}, "float64"],
    ["acdf,jbje,gihb,hfac", {}, {}, "float64"],
    ["acdf,jbje,gihb,hfac,gfac,gifabc,hfac", {}, {}, "float64"],
    ["chd,bde,agbc,hiad,bdi,cgh,agdb", {}, {}, "float64"],
    ["eb,cb,fb->cef", {}, {}, "float64"],
    ["dd,fb,be,cdb->cef", {}, {}, "float64"],
    ["bca,cdb,dbf,afc->", {}, {}, "float64"],
    ["dcc,fce,ea,dbf->ab", {}, {}, "float64"],
    ["a,ac,ab,ad,cd,bd,bc->", {}, {}, "float64"],
)
