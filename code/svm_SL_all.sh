subs="01 12 17 24 27 31 32 33 34 36 37 41"
subs="31 32 33 34 36 37 41"



for s in $subs
do
    echo "================================"
    echo ">> Searchlight Subject $s <<"
    echo "++++++++++++++++++++++++++++++++"
    python svm_searchlight.py $s 2> /dev/null 
done


