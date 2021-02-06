# Purpose: calculate group tstatistic for ROI search results

# first make a group results file. This file will have 12 columns and N-rows,
# where N is equal to the number of ROIs tested

res_stem=../sub-rid0000
res_file=/anal2_roiSL.1D 
subs="01 12 17 24 27 31 32 33 34 36 37 41"

r1=${res_stem}01${res_file}
r2=${res_stem}12${res_file}
r3=${res_stem}17${res_file}
r4=${res_stem}24${res_file}
r5=${res_stem}27${res_file}
r6=${res_stem}31${res_file}
r7=${res_stem}32${res_file}
r8=${res_stem}33${res_file}
r9=${res_stem}34${res_file}
r10=${res_stem}36${res_file}
r11=${res_stem}37${res_file}
r12=${res_stem}41${res_file}

: << STOP
# first sort each file in place using 'sort' command
for s in $subs
do
    f=${res_stem}${s}${res_file} 
    echo $f
    sort $f > __temp
    cat __temp > $f
    rm __temp
done

#STOP

# now "h-stack" the results (third column) of each subject's results and
# save into a single 1D file, AFNI's 1dcat 

col=2 # for the third column, the first is 0, second, 1, etc.
1dcat \
    $r1[$col] \
    $r2[$col] \
    $r3[$col] \
    $r4[$col] \
    $r5[$col] \
    $r6[$col] \
    $r7[$col] \
    $r8[$col] \
    $r9[$col] \
    $r10[$col] \
    $r11[$col] \
    $r12[$col] > ../groupROI.1D

STOP
cat ../groupROI.1D | awk '{ \
sx = 0; ssx = 0;
for(i=1; i<=NF;i++) {sx+=$i; ssx += ($i*$i)}; \
mux=sx/NF; \
mud=mux-muz; \
sqrNF=(NF)^.5; \
var=ssx/(NF-1); \
sd=var^.5; \
t=mud/(sd/sqrNF); \
print mux" "t}' muz=.05
exit 0

cat ../groupROI.1D | awk '{ \
sx = 0; ssx = 0;
for(i=1; i<=NF;i++) {sx+=$i; ssx += ($i*$i)}; \
mux=sx/NF; \
mud=mux-muz; \
y=(sx^2)/NF; \
x=ssx-y; \
z=(NF-1)*NF; \
g=x/y; \
j=g^.5; \
t=mud/j; \
print mux" "t}' muz=.05
exit 0
