import os, sys, ntpath

#print "Current directory is: %s" %os.getcwd()
current_path = "/home/jenny/Desktop/classify/png/"
#print "The dir is: %s" %os.listdir(current_path)
dirname = os.listdir(current_path)
dirname.sort()
#print ntpath.basename(dirname)
new_train_path="/home/jenny/Desktop/classify/train/"
new_validation_path="/home/jenny/Desktop/classify/validation/"
i=0
cnt=1
while i<250:
    print dirname[i]
    filename = dirname[i]
    path = "/home/jenny/Desktop/classify/train/"+filename
    if(os.path.isdir(path)):
        print "exit"
    else:
        print path
        os.mkdir(path)
    path = "/home/jenny/Desktop/classify/validation/"+filename
    if(os.path.isdir(path)):
        print "exit"
    else:
        print path
        os.mkdir(path)
    i=i+1
    j=0
    filename = filename+"/"
    print os.listdir(current_path+filename)
    list_of_file = os.listdir(current_path+filename)
    list_of_file.sort()

    if (len(list_of_file)==80):
        while j<70:
            #new_filename = filename+"/%d.png" %cnt
            new_filename = filename + list_of_file[j]
            print "new_filename %s" %new_filename
            j=j+1
            print current_path+new_filename
            os.rename(current_path+new_filename, new_train_path+new_filename)

        while j<80:
            #new_filename = filename+"/%d.png" %cnt
            new_filename = filename + list_of_file[j]
            print new_filename
            j=j+1
            print current_path+new_filename
            os.rename(current_path+new_filename, new_validation_path+new_filename)
