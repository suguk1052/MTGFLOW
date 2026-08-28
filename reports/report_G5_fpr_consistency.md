# 작업 G-5 보조 — Fisher-tail(B) 정상 FPR 악화 일관성 진단

seed [2024, 2025, 2026, 2027, 2028], fold 24종 × seed = 120 셀. Δ=FPR(B)−FPR(A), Δ>0이면 B가 정상을 더 많이 오탐(악화). threshold=val-normal 95pct.

### 고진폭 정상 FPR (K001/K003/K006) (셀 120개 = fold×seed)

- **B가 A보다 악화(Δ>0) 비율: 56/120 = 47%** (개선 Δ<0: 42, 동일: 22)
- Δ(B−A) mean 0.004 / median 0.000 / std 0.134 / [min -0.282, max 0.626]
- A mean 0.151 → B mean 0.155

- **seed별 평균 Δ**: s2024=-0.004(악화 7/24) / s2025=-0.024(악화 11/24) / s2026=0.039(악화 15/24) / s2027=0.016(악화 11/24) / s2028=-0.007(악화 12/24)
- **fold유형별 평균 Δ**: compositional=-0.023(악화 14/30) / zero-support=0.013(악화 42/90)

- **fold별 평균 Δ (악화 상위 5 / 개선 상위 5, seed 평균)**:
    - ↑악화 012to3_LONO4 (zero-support): Δ=0.143 (n=5)
    - ↑악화 012to3_LONO3 (zero-support): Δ=0.131 (n=5)
    - ↑악화 013to2_LONO4 (zero-support): Δ=0.125 (n=5)
    - ↑악화 023to1_LONO4 (zero-support): Δ=0.118 (n=5)
    - ↑악화 023to1_LONO3 (zero-support): Δ=0.061 (n=5)
    - ↓개선 012to3_LONO1 (zero-support): Δ=-0.227 (n=5)
    - ↓개선 013to2_LONO1 (zero-support): Δ=-0.183 (n=5)
    - ↓개선 123to0_LONO1 (compositional): Δ=-0.139 (n=5)
    - ↓개선 023to1_LONO2 (zero-support): Δ=-0.031 (n=5)
    - ↓개선 012to3_LONO2 (zero-support): Δ=-0.018 (n=5)

### 저진폭 정상 FPR (K002/K004/K005) (셀 120개 = fold×seed)

- **B가 A보다 악화(Δ>0) 비율: 92/120 = 77%** (개선 Δ<0: 25, 동일: 3)
- Δ(B−A) mean 0.096 / median 0.032 / std 0.162 / [min -0.145, max 0.674]
- A mean 0.323 → B mean 0.418

- **seed별 평균 Δ**: s2024=0.102(악화 18/24) / s2025=0.109(악화 20/24) / s2026=0.066(악화 18/24) / s2027=0.120(악화 19/24) / s2028=0.081(악화 17/24)
- **fold유형별 평균 Δ**: compositional=0.093(악화 24/30) / zero-support=0.096(악화 68/90)

- **fold별 평균 Δ (악화 상위 5 / 개선 상위 5, seed 평균)**:
    - ↑악화 013to2_LONO6 (zero-support): Δ=0.430 (n=5)
    - ↑악화 023to1_LONO6 (zero-support): Δ=0.319 (n=5)
    - ↑악화 123to0_LONO6 (compositional): Δ=0.307 (n=5)
    - ↑악화 012to3_LONO6 (zero-support): Δ=0.282 (n=5)
    - ↑악화 013to2_LONO2 (zero-support): Δ=0.201 (n=5)
    - ↓개선 012to3_LONO1 (zero-support): Δ=-0.038 (n=5)
    - ↓개선 123to0_LONO4 (compositional): Δ=-0.025 (n=5)
    - ↓개선 012to3_LONO4 (zero-support): Δ=-0.021 (n=5)
    - ↓개선 013to2_LONO4 (zero-support): Δ=-0.018 (n=5)
    - ↓개선 123to0_LONO1 (compositional): Δ=0.009 (n=5)

> 해석 가이드: 악화 비율이 ~50%에 가깝고 Δ가 작으면 '일관되지만 경미', 특정 seed/fold에 몰리면 '편중'. 저진폭 편중이면 val↔test 진폭 shift(G-3a 잔존 약점)와 동일 축.
