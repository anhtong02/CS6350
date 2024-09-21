# CS6350-HW1
This is a machine learning library developed by Anh Tong for
CS5350/6350 in University of Utah

## Homework 1 - Decision Tree, how to run code on CADE machine.
 When you have the files on Linux CADE machines, MAKE SURE the rp.sh is in DecisionTree folder, DecisionTree folder is in CS6350-HW1 folder. 
1. First, open the rp.sh file, on the first line is: `cd /home/u1269555/Documents/CS6350-HW1/DecisionTree`, change the path according to yours. Whatever you do, last 2 elements (CS6350-HW1 and DecisionTree) have to be there: `cd /a/b/c/CS6350-HW1/DecisionTree` , DO NOT FORGET TO SAVE THE FILE. If you want to remake a new file, then that is okay, as long as it stay in the DecisionTree folder! See how to write the script the file below!
  
2. Then in the terminal, cd to the path you just wrote. So it looks something like this: `cd /home/u1269555/Documents/CS6350-HW1/DecisionTree`. Press enter.
3. Then write: `chmod +x /home/u1269555/Documents/CS6350-HW1/DecisionTree/rp.sh`, change the path you rewrote in step 1, but remember to plug /rp.sh in the end, so yours should look something like this: `chmod +x /a/b/c/CS6350-HW1/DecisionTree/rp.sh`. Press enter.
4. Finally, run the file: `/home/u1269555/Documents/CS6350-HW1/DecisionTree/rp.sh`, once again, change the path, but do not forget last 3 elements. Now yours should look like this : `/a/b/c/CS6350-HW1/DecisionTree/rp.sh`. Press enter.

And it should now run!

Here was what i wrote and it worked:
![image](https://github.com/user-attachments/assets/0a1670c6-ca68-4af1-8cc9-bb09cece8e7d)

### HOW TO WRITE SCRIPT FOR .sh FILE:
This is my script, but you need to change for file directory according to your own! Again, do not forget the last 2 elements should be /CS6350-HW1/DecisionTree, your .sh file should be in DecisionTree folder!
cd /home/u1269555/Documents/CS6350-HW1/DecisionTree

echo "curr directory: $(pwd)"
export PYTHONPATH=$(pwd)

python3 car_predict.py
python3 bank_predict.py

### HOW TO CREATE .sh FILE IN THE FIRST PLACE:
If you have a hard time not knowing how to create shell script in the first place just like me, im here to help!
1. First, open your terminal, cd to your desire directory/ path.
`cd /your/desire/path/CS6350-HW1/DecisionTree`
2. Run this line, name whatever you want your .sh file to be, i choose rp as "run_predict", so it looks like this: `nano rp.sh`, enter!
3. Then you will get to nano editor, paste the script above to your nano editor, do not forget the path you just used in the first step! Control + O to save, then Enter, then control + X to get out of nano editor.
4. Next 2 steps are the same step 3 and 4 from "Homework 1 - Decision Tree, how to run code on CADE machine."

Thanks for reading!
