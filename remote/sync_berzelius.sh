rsync -avzh --exclude-from 'ignore_files.txt' 'ssh -J x_ashsi@berzelius.nsc.liu.se' /home/ashisig/Research/Projects/mmaction2/   x_ashsi@berzelius.nsc.liu.se:/home/x_ashsi/Research/Projects/mmaction2/



rsync -avzhe 'ssh -J x_ashsi@berzelius.nsc.liu.se'   x_ashsi@berzelius.nsc.liu.se:/proj/tinyml_htg_ltu/users/x_ashsi/Research/Data/FineGym/clips/ /home/ashisig/Research/Data/Finegym/videos/