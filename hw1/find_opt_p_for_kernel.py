import os

f = open('timing_p.csv','w')
f.write('m,p,n,mflop,error\n')
f.close()

for p in [16*i for i in range(1,16)]:
    os.system('make -f Makefile_kernel clean')
    os.system('make -f Makefile_kernel P=' + str(p))
    os.system('./ktimer_' + str(p) + ' | grep "PERF" >> timing_p.csv')
