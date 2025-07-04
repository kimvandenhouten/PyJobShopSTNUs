************************************************************************
file with basedata            : md330_.bas
initial value random generator: 2115368279
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  168
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       22        0       22
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           9  16
   3        3          3           5   6  10
   4        3          3          12  14  17
   5        3          3           7  18  20
   6        3          2           8  14
   7        3          2          13  17
   8        3          3          12  13  21
   9        3          1          11
  10        3          3          12  14  18
  11        3          3          13  15  17
  12        3          1          16
  13        3          1          19
  14        3          1          19
  15        3          2          18  20
  16        3          1          20
  17        3          1          21
  18        3          2          19  21
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     7       0    4    0    2
         2     8       0    2    7    0
         3    10       0    1    5    0
  3      1     1       0    9    6    0
         2     2       7    0    6    0
         3    10       6    0    6    0
  4      1     4       4    0    0    9
         2     6       3    0    2    0
         3    10       0    3    0    8
  5      1     3       0    9    5    0
         2     7       0    9    0    9
         3    10       5    0    0    8
  6      1     4       0    5    9    0
         2     6       0    4    5    0
         3    10       6    0    0    6
  7      1     1       4    0    0    7
         2     6       4    0   10    0
         3     9       4    0    0    6
  8      1     5       9    0    0    7
         2     6       8    0    0    5
         3     7       8    0    0    2
  9      1     1       5    0    0    9
         2     6       0    5    0    8
         3     9       5    0    4    0
 10      1     2       3    0   10    0
         2     3       0    5    0   10
         3    10       0    4    7    0
 11      1     2       0    3    0    2
         2     2       4    0    0    3
         3     3       4    0    8    0
 12      1     1       7    0    0    9
         2     2       0    6    0    3
         3    10       7    0    8    0
 13      1     6       6    0    4    0
         2     6       0    5    4    0
         3     7       6    0    0    8
 14      1     1       6    0    0    7
         2     3       0    9    0    5
         3    10       0    5    0    5
 15      1     1       0   10    0    5
         2     4       0    7    0    4
         3     6       5    0    8    0
 16      1     5       5    0    0    9
         2     7       0   10    3    0
         3    10       0    9    0    9
 17      1     4       0    3    0    6
         2     4       0    2    9    0
         3     7       6    0    7    0
 18      1     1       6    0    0    8
         2     6       6    0    4    0
         3     6       5    0    0    3
 19      1     5       0    4    7    0
         2     5       0    4    0    5
         3     6       0    3    8    0
 20      1     6       9    0    0    5
         2     8       5    0    0    4
         3     9       0    9    1    0
 21      1     5      10    0    9    0
         2     7       4    0    6    0
         3     9       0    8    1    0
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   16   16   61   66
************************************************************************
DEADLINES:
jobnr.  deadline
  2       71
  3       152
  4       17
  7       76
  11      136
  12      118
  13      11
  17      101
  20      141
************************************************************************
