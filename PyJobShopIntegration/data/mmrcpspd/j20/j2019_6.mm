************************************************************************
file with basedata            : md339_.bas
initial value random generator: 266955833
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  156
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       35        4       35
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           7  12  13
   3        3          3           5   6  10
   4        3          1           8
   5        3          3           8  12  17
   6        3          1          13
   7        3          3           8  10  17
   8        3          1           9
   9        3          1          11
  10        3          3          11  14  16
  11        3          2          15  21
  12        3          1          15
  13        3          3          17  18  19
  14        3          3          19  20  21
  15        3          1          18
  16        3          3          18  19  21
  17        3          1          20
  18        3          1          20
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     7       0    8    5    0
         2     9       8    0    0    7
         3    10       0    7    0    7
  3      1     2       0    8    4    0
         2     3       3    0    0    5
         3     6       0    7    0    4
  4      1     4       0    7    0    8
         2     8       9    0    0    8
         3     9       7    0    0    7
  5      1     1       6    0    0    9
         2     2       0    5    8    0
         3     6       0    4    7    0
  6      1     2       4    0    0    4
         2     6       3    0    0    4
         3     8       0    2    0    3
  7      1     3       0    3    6    0
         2     3       5    0    0    8
         3     9       5    0    0    1
  8      1     6       0    2    0    8
         2     6       0    4    1    0
         3     8       4    0    0    8
  9      1     3       8    0    0    5
         2     4       0    2    0    4
         3     9       7    0    0    3
 10      1     2       0    3    8    0
         2     2       8    0    8    0
         3     5       3    0    7    0
 11      1     5       0    8    0    9
         2     5       0   10    0    8
         3     9       0    7    0    8
 12      1     4       0    8    0    7
         2     4       1    0    0    8
         3     5       0    8    0    6
 13      1     1       0    5    0    5
         2     3       0    2    0    2
         3     6       7    0    1    0
 14      1     3       0    7   10    0
         2     4       6    0    0    1
         3     8       0    4   10    0
 15      1     6       6    0    7    0
         2     7       0   10    3    0
         3     9       0    5    0    1
 16      1     5       9    0    0    6
         2     6       8    0    4    0
         3     9       7    0    4    0
 17      1     6       6    0    7    0
         2     9       0    6    0    5
         3     9       5    0    3    0
 18      1     2       0    6    0    6
         2     5       0    2    9    0
         3     9       7    0    0    5
 19      1     1       5    0    0    6
         2     5       0    4    0    6
         3     6       1    0    0    4
 20      1     3       2    0    5    0
         2     4       0    7    4    0
         3    10       0    3    3    0
 21      1     4       0   10    8    0
         2     5       0    9    8    0
         3     6      10    0    8    0
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   22   19   67   84
************************************************************************
DEADLINES:
jobnr.  deadline
  2       97
  3       45
  4       8
  5       81
  6       40
  8       151
  11      7
  13      143
  15      132
  18      121
  21      94
************************************************************************
