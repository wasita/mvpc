#!/bin/bash -l

executable=$1
subj=$2
myEmail=andrew.c.connolly@dartmouth.edu
PBS_O_WORKDIR=`pwd`

cat << ENDOFMESSAGE
# declare a name for this job to be my_serial_job
# it is recommended that this name be kept to 16 characters or less
#PBS -N ${executable}_${subj}
# request the queue (enter the possible names, if omitted, default is the
# default)
# this job is going to use the default
#PBS -q default
# request 1 node
#PBS -l nodes=1:ppn=16
# request 0 hours and 15 minutes of wall time
# (Default is 1 hour without this directive)
#PBS -l walltime=00:15:00
# mail is sent to you when the job starts and when it terminates or aborts 
#PBS -m bea
# specify your email address 
#PBS -M ${myEmail}
# By default, PBS scripts execute in your home directory, not the
# directory from which they were submitted. The following line
# places the job in the directory from which the job was submitted.

cd $PBS_O_WORKDIR

# load modules?

module load python/3-Anaconda
# run the program using the relative path 

./${executable} ${subj} 
exit 0
ENDOFMESSAGE
