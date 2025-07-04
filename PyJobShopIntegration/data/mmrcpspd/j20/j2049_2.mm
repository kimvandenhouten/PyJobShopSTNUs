************************************************************************
file with basedata            : md369_.bas
initial value random generator: 1025375284
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  172
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       28       16       28
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2          16  20
   3        3          3           5  10  13
   4        3          3           5  10  13
   5        3          3           6   7   8
   6        3          2           9  21
   7        3          3          11  12  14
   8        3          2          12  15
   9        3          3          14  15  20
  10        3          2          11  19
  11        3          1          17
  12        3          2          18  21
  13        3          2          14  16
  14        3          1          19
  15        3          1          16
  16        3          1          18
  17        3          2          20  21
  18        3          1          19
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     1       0    3    9    9
         2     7       0    2    6    5
         3     9       4    0    2    4
  3      1     1       6    0    8    6
         2     7       0   10    5    5
         3     8       0    5    3    5
  4      1     2       9    0    4    9
         2     4       0    8    3    8
         3     9       4    0    2    8
  5      1     3       7    0    8    7
         2     7       7    0    7    7
         3    10       6    0    5    7
  6      1     1       0    9    8    7
         2     7       0    6    8    7
         3    10       5    0    8    6
  7      1     7       9    0    7   10
         2     7       9    0    8    9
         3    10       9    0    6    8
  8      1     3      10    0    5   10
         2     5       0    7    3    8
         3    10       2    0    2    7
  9      1     3       0    5    2    4
         2     9       7    0    2    3
         3     9       8    0    2    2
 10      1     3       3    0    3    8
         2     5       3    0    2    6
         3     8       0    6    2    4
 11      1     3       0    7    9    6
         2     7       0    7    8    3
         3     9       8    0    8    2
 12      1     1       5    0    6    2
         2     3       0    7    6    2
         3     7       0    4    5    1
 13      1     3       0    5    4    6
         2     5       4    0    3    4
         3     5       0    4    4    5
 14      1     8       9    0    6    9
         2     8       7    0    7    6
         3    10       0    4    6    2
 15      1     5       0    6    2    7
         2     5       5    0    2    8
         3     6       5    0    1    5
 16      1     7       0    9   10    6
         2     7       7    0    9    6
         3     9       0   10    8    6
 17      1     9       0   10    9    6
         2    10       0    6    3    3
         3    10       9    0    5    5
 18      1     1       1    0    9    9
         2     2       1    0    7    7
         3     7       0    4    6    6
 19      1     1       1    0    3    9
         2     4       0    7    3    8
         3     7       0    7    2    5
 20      1     4       0    3    9    5
         2     7       3    0    8    5
         3     9       0    3    7    5
 21      1     4       3    0    8    3
         2    10       0    8    4    2
         3    10       3    0    6    3
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   15    6  120  127
************************************************************************
DEADLINES:
jobnr.  deadline
  3       100
  8       16
  10      47
  13      77
  14      32
  15      110
  17      38
  18      166
  21      112
************************************************************************
