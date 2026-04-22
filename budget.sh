#!/usr/bin/env bash
BUDGET=75.00
TOTAL=0

for f in $(find results -name token_usage.json); do
    [ -f "$f" ] || continue
    cost=$(python3 -c "import json; print(json.load(open('$f'))['total_cost_usd'])")
    TOTAL=$(python3 -c "print($TOTAL + $cost)")
done

REMAINING=$(python3 -c "print(round($BUDGET - $TOTAL, 4))")
PCT=$(python3 -c "print(round($TOTAL / $BUDGET * 100, 2))")

echo "Spent:     \$$TOTAL"
echo "Remaining: \$$REMAINING / \$$BUDGET"
echo "Used:      $PCT%"
