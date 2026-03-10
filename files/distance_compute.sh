#!/bin/bash
TOP="stripped.com_solvated.top"
TRAJ="com_Dry.nc"
BATCH=5000
TOTAL_LINES=$(wc -l < pair_generated.in)
LOG="batch_run.log"

echo "=== Batch run started: $(date) ===" > "$LOG"
echo "Total lines: $TOTAL_LINES" >> "$LOG"
echo "Batch size: $BATCH" >> "$LOG"
echo "" >> "$LOG"

batch_id=0
start=1
while [ $start -le $TOTAL_LINES ]; do
    end=$((start + BATCH - 1))
    if [ $end -gt $TOTAL_LINES ]; then
        end=$TOTAL_LINES
    fi

    id=$(printf "%03d" "$batch_id")
    in_file="cpptraj_batch_${id}.in"

    echo "Starting batch $id (lines $start to $end) at $(date)" >> "$LOG"

    {
        echo "parm $TOP"
        echo "trajin $TRAJ 1 200000 20"
        sed -n "${start},${end}p" pair_generated.in
        # NO create / NO writedata here
    } > "$in_file"

    cpptraj -i "$in_file" >> "$LOG" 2>&1

    echo "Finished batch $id at $(date)" >> "$LOG"
    echo "" >> "$LOG"

    echo "Merging .dat files for batch $id" >> "$LOG"

    # Collect all .dat files created by this batch
    dat_files=( $(ls d_*.dat | sort) )

    if (( ${#dat_files[@]} == 0 )); then
        echo "No .dat files found for batch $id!" >> "$LOG"
        exit 1
    fi

    # Initialize merged_${id}.dat with Frame + first dataset
    cut -f1,2 "${dat_files[0]}" > merged_${id}.dat

    # Append column 2 from each remaining file
    for ((k=1; k<${#dat_files[@]}; k++)); do
        paste merged_${id}.dat <(cut -f2 "${dat_files[$k]}") > temp.dat
        mv temp.dat merged_${id}.dat
    done

    echo "Batch $id merged into merged_${id}.dat" >> "$LOG"

    rm d_*.dat
    echo "Deleted individual .dat files for batch $id" >> "$LOG"

    rm "$in_file"

    start=$((end + 1))
    batch_id=$((batch_id + 1))
done


popd
