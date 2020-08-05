FROM continuumio/miniconda3

COPY . .
RUN apt update

RUN apt install openmpi-bin -y

RUN conda install mpi4py numpy scipy Click PyYAML matplotlib seaborn pillow BTrees persistent transaction

RUN pip install ZODB

RUN python ./setup.py install

RUN boolsi simulate examples/example1.yaml -t 5

RUN boolsi attract examples/example2.yaml

RUN boolsi target examples/example3.yaml

RUN mpiexec -np 2 boolsi simulate examples/example1.yaml -t 5

RUN mpiexec -np 2 boolsi attract examples/example2.yaml

RUN mpiexec -np 2 boolsi target examples/example3.yaml

