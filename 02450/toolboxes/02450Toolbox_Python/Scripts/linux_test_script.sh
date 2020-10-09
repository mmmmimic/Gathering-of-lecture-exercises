#!/bin/bash
# This script is for testing the exercises in 02450 on Ubuntu.
test_python() {
	#~/anaconda3/bin/python3 "$@"
	/usr/bin/python3 "$@"
}
export -f test_python
export PYTHONPATH=$PYTHONPATH:./../Tools

test_python --version
test_python --version >> temp.log 2>&1
cat temp.log
rm temp.log

echo "Installed packages are:"
#~/anaconda3/bin/pip freeze
pip3 freeze

echo "\n\n\n"

run_ex2=true
run_ex3=true
run_ex4=true
run_ex5=true
run_ex6=true
run_ex7=true
run_ex8=true
run_ex9=true
run_ex10=true
run_ex11=true
run_ex12=true


##

if $run_ex2;  then
	echo "\nRunning Exercise 2\n"
	test_python ./ex2_1_1.py
	test_python ./ex2_1_2.py
	test_python ./ex2_1_3.py
	test_python ./ex2_1_4.py
	test_python ./ex2_1_5.py

	test_python ./ex2_2_1.py
	test_python ./ex2_2_2.py

	test_python ./ex2_3_1.py
fi




##
if $run_ex3;  then
	echo "\nRunning Exercise 3\n"
	test_python ./ex3_1_2.py
	test_python ./ex3_1_3.py
	test_python ./ex3_1_4.py
	test_python ./ex3_1_5.py

	test_python ./ex3_2_1.py

	test_python ./ex3_3_1.py
	test_python ./ex3_3_2.py
fi

###
if $run_ex4;  then
	echo "\nRunning Exercise 4\n"
	test_python ./ex4_1_1.py
	test_python ./ex4_1_2.py
	test_python ./ex4_1_3.py
	test_python ./ex4_1_4.py
	test_python ./ex4_1_5.py
	test_python ./ex4_1_6.py
	test_python ./ex4_1_7.py

	test_python ./ex4_2_1.py
	test_python ./ex4_2_2.py

	test_python ./ex4_3_1.py
	test_python ./ex4_3_2.py
	test_python ./ex4_3_3.py
	test_python ./ex4_3_4.py
	test_python ./ex4_3_5.py

	test_python ./ex4_4_1.py
	test_python ./ex4_4_2.py
fi


if $run_ex5;  then
	echo "\nRunning Exercise 5\n"
	test_python ./ex5_1_1.py
	test_python ./ex5_1_2.py
	test_python ./ex5_1_3.py
	test_python ./ex5_1_4.py
	test_python ./ex5_1_5.py
	test_python ./ex5_1_6.py
	test_python ./ex5_1_7.py

	test_python ./ex5_2_1.py
	test_python ./ex5_2_2.py
	test_python ./ex5_2_3.py
	test_python ./ex5_2_4.py
	test_python ./ex5_2_5.py
	test_python ./ex5_2_6.py
fi

if $run_ex6;  then
	echo "\nRunning Exercise 6\n"
	test_python ./ex6_1_1.py
	test_python ./ex6_1_2.py
	
	test_python ./ex6_2_1.py
	
	test_python ./ex6_3_1.py
fi

if $run_ex7;  then
	echo "\nRunning Exercise 7\n"
	test_python ./ex7_1_1.py
	test_python ./ex7_1_2.py
	
	test_python ./ex7_2_3.py
	test_python ./ex7_2_4.py
fi

if $run_ex8;  then
	echo "\nRunning Exercise 8\n"
	test_python ./ex8_1_1.py
	
	test_python ./ex8_2_2.py
	test_python ./ex8_2_5.py
	test_python ./ex8_2_6.py
	
	test_python ./ex8_3_1.py
	test_python ./ex8_3_2.py
fi

if $run_ex9;  then
	echo "\nRunning Exercise 9\n"
	test_python ./ex9_1_1.py
	test_python ./ex9_1_2.py
	
	test_python ./ex9_2_1.py
	test_python ./ex9_2_2.py
	test_python ./ex9_2_3.py
fi

if $run_ex10;  then
	echo "\nRunning Exercise 10\n"
	test_python ./ex10_1_1.py
	test_python ./ex10_1_3.py
	test_python ./ex10_1_5.py
	
	test_python ./ex10_2_1.py
fi

if $run_ex11;  then
	echo "\nRunning Exercise 11\n"
	test_python ./ex11_1_1.py
	test_python ./ex11_1_5.py
	
	test_python ./ex11_2_1.py
	test_python ./ex11_2_2.py
	test_python ./ex11_2_3.py
	test_python ./ex11_2_100.py

	test_python ./ex11_3_1.py
	test_python ./ex11_3_2.py
	
	test_python ./ex11_4_1.py
fi
