import judger_medical as judger
import os
 
imgs = judger.get_file_names()
f = judger.get_output_file_object()
f_tmp = open("tmp_result.txt","w")
 
filenames = []
fout = open("filenames.txt","w")
for img_name in imgs:
    filenames.append(img_name)
    fout.write("%s\n" % img_name)
 
    if len(filenames) >= 10:
        filenames = []
        fout.close()
        os.system("python3 py_new.py")
 
        while True:
            if not os.path.isfile("filenames.txt"):
                break
        fout = open("filenames.txt","w")
         # read single test result
        fin = open("result.txt","r")
        result = fin.read()
 
        # local tmp result
        f_tmp.write(result)

        # write to judge file
        result = result.encode()
        f.write(result)
       
        fin.close()
 
if len(filenames) > 0:
    filenames = []
    fout.close()
    os.system("python3 py_new.py")
 
    while True:
        if not os.path.isfile("filenames.txt"):
            break
    # read single test result
    fin = open("result.txt","r")
    result = fin.read()

    # local tmp result
    f_tmp.write(result)
    
    # write to judge file
    result = result.encode()
    f.write(result)
   
    fout.close()
    fin.close()
 
if os.path.exists("filenames.txt"):
    os.remove("filenames.txt")
if os.path.exists("result.txt"):
    os.remove("result.txt")
 
score, err = judger.judge()
if err is not None:  # in case we failed to judge your submission
    print(err)