

folder=$1
truns=$2

total=369

output=time-$folder.csv
rm -rf $output

#for((i=1;i<=$truns;i++))
for((j=0;j<$total;j++))
do
    ##echo "instance: $folder"
    val="$j"
    #val="$i"
for((i=1;i<=$truns;i++))
#for((j=0;j<$total;j++))
    do
        tval=`grep -Inr TIME $folder-$i/instance_$j | awk '{if($2<299) {print $2} else {print 300} }' `
        val="$val, $tval"
    done
    echo "$val" >> $output
done

#exit

##computing par-10
tpen=3000
opar=par-$folder.csv
rm -rf $opar

for((j=0;j<$total;j++))
do
    val="$j"
    for((i=1;i<$truns;i++))
    do
        tval=`grep -Inr TIME $folder-$i/instance_$j | awk '{if($2<299) {print $2} else {print 3000} }' `
        val="$val, $tval"
    done
    echo "$val" >>$opar
done

exit

oarsub -t allow_classic_ssh -l /nodes=1,walltime=06:00:00 -p "host not like 'stremi-29.reims.grid5000.fr' AND host not like 'stremi-10.reims.grid5000.fr' " "sh ex-par2.sh 1 ex-8-gsize4-4.sh"
oarsub -t allow_classic_ssh -l /nodes=1,walltime=06:00:00 -p "host not like 'stremi-29.reims.grid5000.fr' AND host not like 'stremi-10.reims.grid5000.fr' " "sh ex-par2.sh 1 ex-8-gsize4-5.sh"
oarsub -t allow_classic_ssh -l /nodes=1,walltime=06:00:00 -p "host not like 'stremi-29.reims.grid5000.fr' AND host not like 'stremi-10.reims.grid5000.fr' " "sh ex-par2.sh 1 ex-8-gsize4-6.sh"
oarsub -t allow_classic_ssh -l /nodes=1,walltime=06:00:00 -p "host not like 'stremi-29.reims.grid5000.fr' AND host not like 'stremi-10.reims.grid5000.fr' " "sh ex-par2.sh 1 ex-8-gsize4-10.sh"



oarsub -t allow_classic_ssh -l /nodes=43,walltime=5:30:00 "sh exp-files_t3.sh inst-sat11.txt 1024 ubcsat-nocoop 2 sparrow11 sparrow11-mpi-nocoop "
oarsub -t allow_classic_ssh -l /nodes=43,walltime=5:30:00 "sh exp-files_t3.sh inst-sat11.txt 1024 ubcsat-nocoop 3 sparrow11 sparrow11-mpi-nocoop "
oarsub -t allow_classic_ssh -l /nodes=43,walltime=5:30:00 "sh exp-files_t3.sh inst-sat11.txt 1024 ubcsat-nocoop 4 sparrow11 sparrow11-mpi-nocoop "
oarsub -t allow_classic_ssh -l /nodes=43,walltime=5:30:00 "sh exp-files_t3.sh inst-sat11.txt 1024 ubcsat-nocoop 5 sparrow11 sparrow11-mpi-nocoop "
oarsub -t allow_classic_ssh -l /nodes=43,walltime=5:30:00 "sh exp-files_t3.sh inst-sat11.txt 1024 ubcsat-nocoop 6 sparrow11 sparrow11-mpi-nocoop "
oarsub -t allow_classic_ssh -l /nodes=43,walltime=5:30:00 "sh exp-files_t3.sh inst-sat11.txt 1024 ubcsat-nocoop 7 sparrow11 sparrow11-mpi-nocoop "
oarsub -t allow_classic_ssh -l /nodes=43,walltime=5:30:00 "sh exp-files_t3.sh inst-sat11.txt 1024 ubcsat-nocoop 8 sparrow11 sparrow11-mpi-nocoop "
oarsub -t allow_classic_ssh -l /nodes=43,walltime=5:30:00 "sh exp-files_t3.sh inst-sat11.txt 1024 ubcsat-nocoop 9 sparrow11 sparrow11-mpi-nocoop "
oarsub -t allow_classic_ssh -l /nodes=43,walltime=5:30:00 "sh exp-files_t3.sh inst-sat11.txt 1024 ubcsat-nocoop 10 sparrow11 sparrow11-mpi-nocoop "


oarsub -t allow_classic_ssh -l /nodes=22,walltime=5:30:00 "sh exp-files_t4.sh inst-sat11.txt 512 ubcsat-nocoop 6 sparrow11 sparrow11-mpi-nocoop "
oarsub -t allow_classic_ssh -l /nodes=22,walltime=5:30:00 "sh exp-files_t4.sh inst-sat11.txt 512 ubcsat-nocoop 7 sparrow11 sparrow11-mpi-nocoop "
oarsub -t allow_classic_ssh -l /nodes=22,walltime=5:30:00 "sh exp-files_t4.sh inst-sat11.txt 512 ubcsat-nocoop 9 sparrow11 sparrow11-mpi-nocoop "


oarsub -t allow_classic_ssh -l /nodes=22,walltime=5:30:00 "sh exp-files_t3.sh inst-sat11.txt 512 ubcsat-nocoop 1 sparrow11 sparrow11-mpi-nocoop "
oarsub -t allow_classic_ssh -l /nodes=22,walltime=5:30:00 "sh exp-files_t3.sh inst-sat11.txt 512 ubcsat-nocoop 2 sparrow11 sparrow11-mpi-nocoop "
oarsub -t allow_classic_ssh -l /nodes=22,walltime=5:30:00 "sh exp-files_t3.sh inst-sat11.txt 512 ubcsat-nocoop 3 sparrow11 sparrow11-mpi-nocoop "
oarsub -t allow_classic_ssh -l /nodes=22,walltime=5:30:00 "sh exp-files_t3.sh inst-sat11.txt 512 ubcsat-nocoop 4 sparrow11 sparrow11-mpi-nocoop "
oarsub -t allow_classic_ssh -l /nodes=22,walltime=5:30:00 "sh exp-files_t3.sh inst-sat11.txt 512 ubcsat-nocoop 5 sparrow11 sparrow11-mpi-nocoop "
oarsub -t allow_classic_ssh -l /nodes=22,walltime=5:30:00 "sh exp-files_t3.sh inst-sat11.txt 512 ubcsat-nocoop 8 sparrow11 sparrow11-mpi-nocoop "
oarsub -t allow_classic_ssh -l /nodes=22,walltime=5:30:00 "sh exp-files_t3.sh inst-sat11.txt 512 ubcsat-nocoop 10 sparrow11 sparrow11-mpi-nocoop "

sh ex-par5.sh 1 ex-craft-sparrow11-4p.sh
sh ex-par2.sh 1 ex-craft-sparrow11-8p.sh
sh ex-craft-sparrow11-16p.sh




##sh exp-files_tp3.sh inst-sat11.txt 256 ubcsat-nocoop 1 "sparrow11 paws adaptg2wsat g2wsat adaptnovelty+ novelty+ novelty++ novelty+p rnovelty" def1-mpi-nocoop NODES-11-1s


echo "sh exp-files_t3-craft.sh crafted-sat.txt 256 ubcsat-coop 5 sparrow11 sparrow11-mpi-coop " > ex-256-1.sh
echo "sh exp-files_t3-craft.sh crafted-sat.txt 256 ubcsat-coop 6 sparrow11 sparrow11-mpi-coop " >> ex-256-1.sh
echo "sh exp-files_t3-craft.sh crafted-sat.txt 256 ubcsat-coop 7 sparrow11 sparrow11-mpi-coop " >> ex-256-1.sh

oarsub -t allow_classic_ssh -l /nodes=11,walltime=13:50:00 "sh ex-256-1.sh"


echo "sh exp-files_t3-craft.sh crafted-sat.txt 256 ubcsat-coop 8 sparrow11 sparrow11-mpi-coop " > ex-256-2.sh
echo "sh exp-files_t3-craft.sh crafted-sat.txt 256 ubcsat-coop 9 sparrow11 sparrow11-mpi-coop " >> ex-256-2.sh
echo "sh exp-files_t3-craft.sh crafted-sat.txt 256 ubcsat-coop 10 sparrow11 sparrow11-mpi-coop " >> ex-256-2.sh

oarsub -t allow_classic_ssh -l /nodes=11,walltime=13:45:00 "sh ex-256-2.sh"


echo "sh exp-files_t3-craft.sh crafted-sat.txt 128 ubcsat-coop 5 sparrow11 sparrow11-mpi-coop " > ex-128.sh
echo "sh exp-files_t3-craft.sh crafted-sat.txt 128 ubcsat-coop 6 sparrow11 sparrow11-mpi-coop " >> ex-128.sh
echo "sh exp-files_t3-craft.sh crafted-sat.txt 128 ubcsat-coop 7 sparrow11 sparrow11-mpi-coop " >> ex-128.sh

oarsub -t allow_classic_ssh -l /nodes=6,walltime=13:45:00 "sh ex-128.sh"


echo "sh exp-files_t3-craft.sh crafted-sat.txt 128 ubcsat-coop 8 sparrow11 sparrow11-mpi-coop " > ex-128-2.sh
echo "sh exp-files_t3-craft.sh crafted-sat.txt 128 ubcsat-coop 9 sparrow11 sparrow11-mpi-coop " >> ex-128-2.sh
echo "sh exp-files_t3-craft.sh crafted-sat.txt 128 ubcsat-coop 10 sparrow11 sparrow11-mpi-coop " >> ex-128-2.sh


oarsub -t allow_classic_ssh -l /nodes=6,walltime=13:45:00 "sh ex-128-2.sh"


echo "sh exp-files_t3-craft.sh crafted-sat.txt 64 ubcsat-coop 5 sparrow11 sparrow11-mpi-coop " > ex-64-1.sh
echo "sh exp-files_t3-craft.sh crafted-sat.txt 64 ubcsat-coop 6 sparrow11 sparrow11-mpi-coop " >> ex-64-1.sh
echo "sh exp-files_t3-craft.sh crafted-sat.txt 64 ubcsat-coop 7 sparrow11 sparrow11-mpi-coop " >> ex-64-1.sh


oarsub -t allow_classic_ssh -l /nodes=3,walltime=13:45:00 "sh ex-64-1.sh"


echo "sh exp-files_t3-craft.sh crafted-sat.txt 64 ubcsat-coop 8 sparrow11 sparrow11-mpi-coop " > ex-64-2.sh
echo "sh exp-files_t3-craft.sh crafted-sat.txt 64 ubcsat-coop 9 sparrow11 sparrow11-mpi-coop " >> ex-64-2.sh
echo "sh exp-files_t3-craft.sh crafted-sat.txt 64 ubcsat-coop 10 sparrow11 sparrow11-mpi-coop " >> ex-64-2.sh


oarsub -t allow_classic_ssh -l /nodes=3,walltime=13:45:00 "sh ex-64-2.sh"


echo "sh exp-files_t3-craft.sh crafted-sat.txt 32 ubcsat-coop 1 sparrow11 sparrow11-mpi-coop " > ex-32-1.sh
echo "sh exp-files_t3-craft.sh crafted-sat.txt 32 ubcsat-coop 2 sparrow11 sparrow11-mpi-coop " >> ex-32-1.sh
echo "sh exp-files_t3-craft.sh crafted-sat.txt 32 ubcsat-coop 3 sparrow11 sparrow11-mpi-coop " >> ex-32-1.sh


oarsub -t allow_classic_ssh -l /nodes=2,walltime=13:45:00 "sh ex-32-1.sh"


echo "sh exp-files_t3-craft.sh crafted-sat.txt 32 ubcsat-nocoop 1 sparrow11 sparrow11-mpi-nocoop " > ex-32-2.sh
echo "sh exp-files_t3-craft.sh crafted-sat.txt 32 ubcsat-nocoop 2 sparrow11 sparrow11-mpi-nocoop " >> ex-32-2.sh
echo "sh exp-files_t3-craft.sh crafted-sat.txt 32 ubcsat-nocoop 3 sparrow11 sparrow11-mpi-nocoop " >> ex-32-2.sh


oarsub -t allow_classic_ssh -l /nodes=2,walltime=13:45:00 "sh ex-32-2.sh"













