"""Small exact controls: cross terms, rounding rank lift, and a valid restricted exception."""
from pathlib import Path
from fractions import Fraction as F
import importlib.util,json,random,struct
import numpy as np
# Import definitions without rerunning the main sweep.
ROOT=Path(__file__).resolve().parent
source=(ROOT/'consolidation.py').read_text();definitions=source[:source.index("protocol={'shape'")];ns={'__file__':str(ROOT/'consolidation.py')};exec(compile(definitions,str(ROOT/'consolidation.py'),'exec'),ns)
f=np.float32;outer=ns['outer_sum'];rank=ns['rank'];fraction=ns['to_fraction'];mm=ns['mismatch'];moment=ns['moment_step'];round32=ns['round32'];OUT=ROOT/'results'
# Exact within-step sum square: two microbatches are accumulated before Adam's square.
A=[[F(1),F(-1)],[F(2),F(1)],[F(-1),F(2)]];B=[[F(2),F(1)],[F(-1),F(2)],[F(3),F(-2)]]
G=outer(A,B);full=[[v*v for v in row] for row in G]
omit=[[sum((A[i][k]*B[j][k])**2 for k in range(2)) for j in range(3)] for i in range(3)]
cross=[[F(2)*A[i][0]*A[i][1]*B[j][0]*B[j][1] for j in range(3)] for i in range(3)]
assert all(full[i][j]==omit[i][j]+cross[i][j] for i in range(3) for j in range(3))
U,V=ns['lifted'](A,B);assert outer(U,V)==full
cross_result={'shape':[3,3],'accumulated_microbatch_rank':2,'square_of_sum':[[str(x) for x in row] for row in full],'sum_of_individual_squares':[[str(x) for x in row] for row in omit],'missing_cross_term':[[str(x) for x in row] for row in cross],'different_entries_without_cross_terms':sum(full[i][j]!=omit[i][j] for i in range(3) for j in range(3)),'exact_lift_identity':True,'scalar_cancellation_control':{'microbatch_gradients':[2,-2],'correct_square_of_sum':0,'wrong_sum_of_squares':8},'gradient_average_note':'For a mean over two microbatches divide BOTH full square and cross term by4; cross terms do not disappear.'}
(OUT/'cross_microbatch.json').write_text(json.dumps(cross_result,indent=2))
# A rounded outer product need not have rank1 over the exact real values of its FP32 entries.
rng=random.Random(2026);a=np.array([ns['asf'](0x3f000000|rng.getrandbits(23)) for _ in range(6)],dtype=np.float32)[:,None];b=np.array([ns['asf'](0x3f000000|rng.getrandbits(23)) for _ in range(6)],dtype=np.float32)[:,None]
real=outer(fraction(a),fraction(b));g=ns['fp32_gradient'](a,b)
assert rank(real)==1 and rank(fraction(g))>1
for name,data in [('A',a),('B',b),('gradient',g)]:data.astype('<f4').tofile(OUT/f'rounded_outer.{name}.f32')
round_rank={'shape':[6,6],'rank_real_product':rank(real),'rank_fp32_product_over_exact_rationals':rank(fraction(g)),'all_products_equal_once_rounded_real':bool(np.array_equal(g.view(np.uint32),ns['rounded'](real).view(np.uint32))),'note':'Full exact matrix rank does NOT imply incompressible data: these same A/B plus the rounded-product decoder still encode it compactly. It obstructs plain exact-real low-rank UV equality, not all nonlinear encodings.'}
(OUT/'rounded_outer_rank.json').write_text(json.dumps(round_rank,indent=2))
# Constructive fixed-basis exception. Scaling by powers of two commutes with normal-range FP32 rounding.
rows=np.array([-3,-2,-1,0,1,2,3,4],dtype=np.int8);cols=np.array([-4,-3,-2,-1,0,1,2,3],dtype=np.int8);exp=rows[:,None].astype(np.int16)+cols[None,:].astype(np.int16)
m=np.zeros((8,8),np.float32);v=m.copy();sm=np.zeros((1,1),np.float32);sv=sm.copy();rng=random.Random(991);trajectory=[]
for t in range(1,65):
 c=f(rng.randrange(-1000,1001)/113.0)
 g=np.ldexp(np.full((8,8),c,dtype=np.float32),exp).astype(np.float32)
 m,v=moment(m,v,g);sm,sv=moment(sm,sv,np.array([[c]],dtype=np.float32))
 dm=np.ldexp(np.full((8,8),sm[0,0],np.float32),exp).astype(np.float32);dv=np.ldexp(np.full((8,8),sv[0,0],np.float32),2*exp).astype(np.float32)
 assert np.array_equal(m.view(np.uint32),dm.view(np.uint32)) and np.array_equal(v.view(np.uint32),dv.view(np.uint32))
 assert np.isfinite(m).all() and np.isfinite(v).all() and np.all(v>np.finfo(np.float32).tiny)
 trajectory.append({'step':t,'input_scalar_word':ns['word'](c),'m_scalar_word':ns['word'](sm[0,0]),'v_scalar_word':ns['word'](sv[0,0]),'all128_moment_words_equal':True})
np.stack([m,v]).astype('<f4').tofile(OUT/'fixed_power2.final_m_v.f32')
exception={'shape':[8,8],'steps':64,'all_raw_moment_words_equal':True,'words_compared':64*8*8*2,'moment_scalar_payload_bytes':8,'row_column_exponent_bytes':16,'shape_step_and_format_metadata_bytes':16,'four_fp32_coefficient_bytes':16,'persistent_representation_and_coefficients_bytes':56,'raw_dense_m_v_bytes':512,'not_counted_as_measured_peak':'This is a representational payload audit, not a low-memory runtime implementation; Python temporaries/stack and reconstruction workspace are not measured. No weights/forward/backward training modeled.','restrictions':'Every gradient has exactly the same separable power-of-two spatial pattern times one scalar; no overflow/subnormal violation; arbitrary changing training gradients need not satisfy this.','trajectory':trajectory}
(OUT/'fixed_power2_exception.json').write_text(json.dumps(exception,indent=2))
# Check direct rational rounding against exactly representable words plus explicit halfway ties.
check_words=[0,1,0x007fffff,0x00800000,0x3f000000,0x3f800000,0x3f800001,0x7f7fffff,0x80000001,0xbf800000]
for w in check_words:assert ns['word'](round32(F(float(ns['asf'](w)))))==w
one=F(1);ulp=F(1,2**23);assert ns['word'](round32(one+ulp/2))==0x3f800000;assert ns['word'](round32(one+3*ulp/2))==0x3f800002
(OUT/'direct_rounding_tests.json').write_text(json.dumps({'exact_word_roundtrips':check_words,'halfway_even_tests':2,'all_passed':True},indent=2))
print(json.dumps({'cross_term_differences':cross_result['different_entries_without_cross_terms'],'outer_rank_before_after':[round_rank['rank_real_product'],round_rank['rank_fp32_product_over_exact_rationals']],'fixed_basis_steps':64,'fixed_basis_words_exact':exception['words_compared'],'rounding_tests_passed':True},indent=2))
