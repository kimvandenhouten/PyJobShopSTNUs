************************************************************************
file with basedata            : md329_.bas
initial value random generator: 21479304
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  160
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       25        8       25
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           8  12  16
   3        3          3           5   9  12
   4        3          3           6  13  21
   5        3          3           7  11  16
   6        3          2           7   8
   7        3          3          10  14  17
   8        3          1          17
   9        3          3          10  15  17
  10        3          1          18
  11        3          2          15  19
  12        3          2          14  21
  13        3          2          14  16
  14        3          1          15
  15        3          1          18
  16        3          1          19
  17        3          2          18  19
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
  2      1     1       0    1    2    0
         2     7       9    0    0    3
         3     9       4    0    0    3
  3      1     2       0    6    9    0
         2     3       0    5    0    6
         3     9       0    3    8    0
  4      1     5       8    0    3    0
         2     7       7    0    0    9
         3     7       0    3    0   10
  5      1     2      10    0    0    7
         2     8      10    0    0    5
         3     9       0    3    1    0
  6      1     4       3    0    0    6
         2     5       2    0    2    0
         3     9       2    0    0    2
  7      1     2       0    5    8    0
         2     7       0    2    0    3
         3    10       5    0    0    1
  8      1     5       0    7    8    0
         2     7       0    7    5    0
         3     8       0    6    4    0
  9      1     2       0    9    7    0
         2     8       7    0    7    0
         3     8       7    0    0   10
 10      1     3       4    0    0    4
         2     4       0    7    6    0
         3     4       2    0    8    0
 11      1     2       0    8    0    9
         2     6       4    0    9    0
         3    10       0    3    7    0
 12      1     3       0    7    0    9
         2     3       5    0    7    0
         3     4       5    0    3    0
 13      1     4       0    6    6    0
         2     6       0    6    0   10
         3     9       0    5    0   10
 14      1     3       0    6    0    7
         2     8       9    0    0    6
         3     8       8    0    0    7
 15      1     7       5    0    0    8
         2     7       0    8    0    2
         3     8       0    6    7    0
 16      1     4       0    9    3    0
         2     6       0    8    0    5
         3     9       0    7    0    2
 17      1     5       0    2    6    0
         2     5       3    0    0    9
         3     9       0    2    0    8
 18      1     3       0    4    5    0
         2     7       5    0    5    0
         3    10       5    0    0    3
 19      1     1       6    0    0    4
         2     2       6    0    3    0
         3     5       6    0    2    0
 20      1     1       6    0    9    0
         2     5       5    0    0    4
         3     7       0    4    9    0
 21      1     1       0    9    3    0
         2     5       5    0    0    5
         3     8       0    3    3    0
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   13   13   55   64
************************************************************************
DEADLINES:
jobnr.  deadline
  6       58
  9       128
  16      116
  20      25
************************************************************************
