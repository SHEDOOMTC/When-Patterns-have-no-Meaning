#!/bin/bash

# User-defined residue list with ranges
RES_INPUT="63-73,97-98,100-112,117-118,138,169-170,172-178,180-181,218,221-235,258,261-273,287-292,366-367,369-378"


# Atom types
IFS=',' read -r -a ATOMS <<< "N,CA,C,O,CB,CG,CD,CE,CZ,NE,NE1,NE2,NZ,OE1,OE2,OD1,OD2,SG,SD,CG1,CG2,CD1,CD2,ND1,ND2,NH1,NH2,OG,OG1,OH,CH2"

# Output file
OUT="pair_generated.in"
> "$OUT"

# --- Expand residue list ---
RES=()

IFS=',' read -r -a ITEMS <<< "$RES_INPUT"
for item in "${ITEMS[@]}"; do
  if [[ "$item" == *"-"* ]]; then
    # It's a range
    start=${item%-*}
    end=${item#*-}
    for ((r=start; r<=end; r++)); do
      RES+=("$r")
    done
  else
    # Single residue
    RES+=("$item")
  fi
done

# --- Generate unique pairs ---
for ((i=0; i<${#RES[@]}; i++)); do
  for ((j=i+1; j<${#RES[@]}; j++)); do   # <-- THIS is the correction

    Ri=${RES[$i]}
    Rj=${RES[$j]}

    for ((a=0; a<${#ATOMS[@]}; a++)); do
      for ((b=a; b<${#ATOMS[@]}; b++)); do

        Ai=${ATOMS[$a]}
        Bj=${ATOMS[$b]}

        echo "distance d_${Ri}_${Ai}_${Rj}_${Bj} :${Ri}@${Ai} :${Rj}@${Bj} out d_${Ri}_${Ai}_${Rj}_${Bj}.dat" >> "$OUT"

      done
    done

  done
done
```
