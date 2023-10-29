for i in mol*.gro
do
        fname="${i%.*}"
        gmx editconf -f $i -c -o ${fname}_centered.gro
done


for i in mol*centered.gro
do
        fname="${i%.*}"
        gmx insert-molecules -f $i -ci water.gro -nmol 10 -o ${fname}_solv.gro
done
