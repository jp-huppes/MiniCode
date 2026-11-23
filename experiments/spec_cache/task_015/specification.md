## Objective
Implementar a função `safe_div(a, b, default=0.0)` que retorna `a / b` se `b != 0`, caso contrário retorna `default`.

## Parameters
- a
- b
- default

## Acceptance Criteria
- **AC-1**: A função deve retornar `a / b` quando `b` é diferente de zero.
- **AC-2**: A função deve retornar `default` quando `b` é igual a zero.

## Tests
- **AC-1**: test_ac_1_division_by_non_zero
- **AC-2**: test_ac_2_division_by_zero_returns_default