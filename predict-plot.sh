series=(1 100 200 216 218 220 225 230 235 240 300 400 500 600 700 800 900)

for num in ${series[@]}; do
	bash predict-web.sh $num
done
