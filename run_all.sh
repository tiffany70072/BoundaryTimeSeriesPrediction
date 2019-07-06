data_list="25 50 75 half uniform output ub lof o3"
place_list="taxi ld bj parking"
result=result_0403

for place in $place_list; do
	for data in $data_list; do
		#CUDA_VISIBLE_DEVICES=2 python3 read_data.py $data 
		CUDA_VISIBLE_DEVICES=2 python3 baseline_rnn.py $data $place _ $result

		CUDA_VISIBLE_DEVICES=2 python3 main_two_rebuild.py $data yes gru $result $place 0
		CUDA_VISIBLE_DEVICES=2 python3 main_two_rebuild.py $data clip gru $result $place 0
		CUDA_VISIBLE_DEVICES=2 python3 main_two_rebuild.py $data mse gru $result $place 0
		CUDA_VISIBLE_DEVICES=2 python3 main_two_rebuild.py $data mse dgru $result $place 0
		CUDA_VISIBLE_DEVICES=2 python3 main_two_rebuild.py $data mse gru2w $result $place 0
		CUDA_VISIBLE_DEVICES=2 python3 main_two_rebuild.py $data mse gru2w_update $result $place 0
		CUDA_VISIBLE_DEVICES=2 python3 main_two_rebuild.py $data mse gru2w_all $result $place 0

		#CUDA_VISIBLE_DEVICES=2 python3 baseline_rnn.py $data $place
		#CUDA_VISIBLE_DEVICES=2 python3 main_two_rebuild.py $data mse gru2w $result $place 1
		#CUDA_VISIBLE_DEVICES=2 python3 main_two_rebuild.py $data mse gru2w $result $place 2
		CUDA_VISIBLE_DEVICES=2 python3 main_two_rebuild.py $data mse gru2w $result $place 3
		CUDA_VISIBLE_DEVICES=2 python3 main_two_rebuild.py $data mse gru2w_update $result $place 3
		CUDA_VISIBLE_DEVICES=2 python3 main_two_rebuild.py $data mse gru2w_all $result $place 3	
	done
done

