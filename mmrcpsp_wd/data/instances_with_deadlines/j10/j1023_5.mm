************************************************************************
file with basedata            : mm23_.bas
initial value random generator: 1942692585
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  88
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       14        6       14
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           5
   3        3          3           9  10  11
   4        3          3           7   8  11
   5        3          2           6  10
   6        3          1           9
   7        3          1           9
   8        3          1          10
   9        3          1          12
  10        3          1          12
  11        3          1          12
  12        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     2       8    0    4    6
         2     7       7    0    2    5
         3     7       0    6    3    6
  3      1     3       2    0    8   10
         2     6       0    5    6    4
         3     8       0    4    5    1
  4      1     3       2    0    5    7
         2     9       0    5    4    6
         3     9       0    4    5    5
  5      1     2       0    7    6    4
         2     9       0    7    5    4
         3    10       4    0    4    2
  6      1     1       0    9    4    5
         2     2       0    9    3    4
         3     9       0    8    2    2
  7      1     3       0    4    6    9
         2     5       0    4    6    8
         3    10       0    3    4    8
  8      1     8       0    8    8    5
         2     9       0    8    6    4
         3    10       0    8    5    3
  9      1     3       0    3   10    7
         2     3       0    4    8    7
         3     9       7    0    5    5
 10      1     3       0    2    6    5
         2     5       0    1    5    4
         3     7       4    0    4    4
 11      1     2       0    2    4    8
         2     5       0    1    3    8
         3     9       7    0    3    7
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
    8   19   54   59
************************************************************************
DEADLINES:
jobnr.  deadline
  3       51
  7       10
  9       82
************************************************************************
