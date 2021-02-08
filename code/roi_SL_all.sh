subs="01 12 17 24 27 31 32 33 34 36 37 41"

for s in $subs
do
    echo "================================"
    echo ">> ROI Searchlight Subject $s <<"
    echo "++++++++++++++++++++++++++++++++"
    python roi_searchlight.py $s 2> /dev/null 
done


