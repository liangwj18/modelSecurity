with open("output_gen_post.txt",'r') as f:
    lines = f.readlines()
    for i in range(0,len(lines),2):
        line = lines[i]
        x = line.split("\n")[0]
        x = x.split(" ")
        a,b,c = float(x[0]),float(x[1]),float(x[2])

        line = lines[i+1]
        x = line.split("\n")[0]
        x = x.split(" ")
        e,f,g = float(x[0]),float(x[1]),float(x[2])

        x, y = (a+f)/2, (e+b)/2

        s = x + y
        x/=s
        y/=s
        print(round(x,4),'&',round((c+g)/2,4))