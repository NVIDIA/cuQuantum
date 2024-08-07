# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cuquantum


# We include torch tests here unconditionally, and use pytest deselect to
# exclude them if torch is not present.
backend_names = (
    "numpy",
    "cupy",
    "torch-cpu",
    "torch-gpu",
)


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
    "ii->",
    "jii->ij",
    "ij,jb,ah",
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
    # next network might request huge workspace for a non-optimized/sequential path, in such case it should be waived
    ["ao,bp,cq,dr,es,ft,gu,hv,iw,jx,ky,lz,mA,nB,oP,pO,qN,rM,sL,tK,uJ,vI,wH,xG,yF,zE,AD,BC,DC,FC,PC,CQ,FD,ID,DR,JE,KE,PE,ES,GF,FT,LG,NG,GU,IH,JH,MH,HV,KI,IW,KJ,JX,KY,NL,OL,LZ,NM,OM,MÀ,NÁ,PO,OÂ,PÃ,RQ,TQ,ÃQ,QÄ,TR,WR,RÅ,XS,YS,ÃS,SÆ,UT,TÇ,ZU,ÁU,UÈ,WV,XV,ÀV,VÉ,YW,WÊ,YX,XË,YÌ,ÁZ,ÂZ,ZÍ,ÁÀ,ÂÀ,ÀÎ,ÁÏ,ÃÂ,ÂÐ,ÃÑ,Äß,ÅÞ,ÆÝ,ÇÜ,ÈÛ,ÉÚ,ÊÙ,ËØ,Ì×,ÍÖ,ÎÕ,ÏÔ,ÐÓ,ÑÒ->", {}, {}, "float64"],  # QAOA MaxCut
    ["ab,bc->ac", {'compute_type': cuquantum.ComputeType.COMPUTE_64F}, {}, "complex128"],
    ["ab,bc->ac", {'compute_type': cuquantum.ComputeType.COMPUTE_3XTF32}, {}, "complex64"],

    # CuPy large TN tests
    ["a,b,c->abc", {}, {}, "float64"],
    ["acdf,jbje,gihb,hfac", {}, {}, "float64"],
    ["acdf,jbje,gihb,hfac,gfac,gifabc,hfac", {}, {}, "float64"],
    ["chd,bde,agbc,hiad,bdi,cgh,agdb", {"blocking": "auto"}, {}, "float64"],
    ["eb,cb,fb->cef", {}, {}, "float64"],
    ["dd,fb,be,cdb->cef", {}, {}, "float64"],
    ["bca,cdb,dbf,afc->", {}, {}, "float64"],
    ["dcc,fce,ea,dbf->ab", {}, {}, "float64"],
    ["a,ac,ab,ad,cd,bd,bc->", {}, {}, "float64"],
)


# the expression here should be
#   - a sequence of [decomposition_expression, input_tensor_shapes as a list of tuple]
tensor_decomp_expressions = (
    ('ab->ax,xb', [(8, 8)]),
    ('ab->ax,bx', [(8, 8)]),
    ('ab->xa,xb', [(8, 8)]),
    ('ab->xa,bx', [(8, 8)]),
    ('ab->ax,xb', [(6, 8)]),
    ('ab->ax,bx', [(6, 8)]),
    ('ab->xa,xb', [(6, 8)]),
    ('ab->xa,bx', [(6, 8)]),
    ('ab->ax,xb', [(8, 6)]),
    ('ab->ax,bx', [(8, 6)]),
    ('ab->xa,xb', [(8, 6)]),
    ('ab->xa,bx', [(8, 6)]),
    ('abcd->cxa,bdx', [(2, 3, 4, 5)]),
    ('abcd->cax,bdx', [(2, 3, 4, 5)]),
    ('mnijk->jny,kmyi', [(2, 9, 3, 3, 4)]),
)


# the expression here should be
#   - a sequence of [gate_decomposition_expression, input_tensor_shapes as a list of tuple]
gate_decomp_expressions = (
    ('ijk,klm,jlpq->ipk,kqm', [(2, 2, 2), (2, 2, 2), (2, 2, 2, 2)]),
    ('ijk,klm,jlpq->kpi,qmk', [(2, 2, 2), (2, 2, 2), (2, 2, 2, 2)]),
    ('ijk,klm,jlpq->pki,mkq', [(2, 2, 2), (2, 2, 2), (2, 2, 2, 2)]),
    ('sOD,DdNr,ROrsq->KR,qKdN', [(2, 4, 2), (2, 3, 4, 2), (5, 4, 2, 2, 2)]),
    ('beQ,cey,cbJj->Je,jQey', [(3, 5, 4), (2, 5, 7), (2, 3, 4, 4)])
)


# the expression here can be
#   - a string as a standard contract and decompose expression
#   - a list of [contract decompose expression, network options, optimize options, kwargs]
contract_decompose_expr = (
    'ea,fb,abcd,gc,hd->exf,gxh',
    'ij,jk,kl->ix,lx',
    'ijk,klm,jlpq->ipk,kqm',
    'abcd,cdef->axb,fex',
    'abcd,cdef->axf,bex',
    ['sOD,DdNr,ROrsq->KR,qKdN', {'blocking': 'auto'}, {}, {}],
    'beQ,cey,cbJj->Je,jQey',
    'ijlm,jqr,lqsn->imx,xrsn',
    ['ijk,klm,jlpq->ipk,kqm', {}, {}, {'return_info': False}],
    ['sOD,DdNr,ROrsq->KR,qKdN', {'device_id':0}, {'slicing': {'min_slices': 4}}, {'return_info': False}],
    ['ea,fb,abcd,gc,hd->exf,gxh', {'device_id':0}, {'path': [(2,4), (0,3), (0,2), (0,1)]}, {}],
)
