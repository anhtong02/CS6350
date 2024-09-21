# CS6350-HW1
This is a machine learning library developed by Anh Tong for
CS5350/6350 in University of Utah

## Homework 1 - Decision Tree, how to run code on CADE machine.
1. Make sure you know your path to the folder, because you need it for step 2. Whatever your path is, last 2 elements (CS6350-HW1 and DecisionTree) have to be there: `/your/own/path/CS6350-HW1/DecisionTree`
2. Then in the terminal, cd to the path from step 1. So it looks like this: `cd /your/own/path/CS6350-HW1/DecisionTree`, but last 2 elements (CS6350-HW1 and DecisionTree) always the same. Press enter.
3. Then write: `chmod +x rp3.sh`. Press enter.
4. Finally, run the file: `/your/own/path/CS6350-HW1/DecisionTree/rp3.sh`, once again, change the path, but do not forget last 3 elements. Press enter.

And it should now run!

Here was what i wrote and it worked:
![image](https://github.com/user-attachments/assets/88132f57-5bb5-470f-b5bd-c0f0c4f4ac82)
Even though below the step 4 line, it tells me "No such file or directory" but the code still runs!

If relative path does not work, I have direction on how to make it to absolute path, as well as recreate the script below.
### HOW TO WRITE SCRIPT (ABSOLUTE PATH) FOR .sh FILE: 
This is my script, but you need to change for file directory according to your own! Again, do not forget the last 2 elements should be /CS6350-HW1/DecisionTree, your .sh file should be in DecisionTree folder!

```bash
cd /home/u1269555/Documents/CS6350-HW1/DecisionTree

echo "curr directory: $(pwd)"
export PYTHONPATH=$(pwd)

python3 car_predict.py
python3 bank_predict.py
```

### HOW TO CREATE .sh FILE IN THE FIRST PLACE:
If you have a hard time not knowing how to create shell script in the first place just like me, im here to help!
1. First, open your terminal, cd to your desire directory or path:
`cd /your/desire/path/CS6350-HW1/DecisionTree`
2. Run this line, name whatever you want your .sh file to be, i choose rp as "run_predict", so it looks like this: `nano your_desire_filename.sh`, then enter.
3. Then you will get to nano editor, paste the script above to your nano editor, do not forget to change the path you just used in the first step! Next, Control + O, then Enter to save, then control + X to get out of nano editor.
4. Next 2 steps are the same step 3 and 4 from "Homework 1 - Decision Tree, how to run code on CADE machine."

Thanks for reading!
