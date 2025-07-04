************************************************************************
file with basedata            : md333_.bas
initial value random generator: 1670268929
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  153
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       24        4       24
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           7   8   9
   3        3          2           6   9
   4        3          3           5   8  12
   5        3          2          11  19
   6        3          3          10  11  12
   7        3          2          11  19
   8        3          3          15  16  21
   9        3          3          19  20  21
  10        3          1          20
  11        3          2          13  16
  12        3          2          13  17
  13        3          2          14  15
  14        3          2          18  21
  15        3          1          18
  16        3          1          17
  17        3          1          18
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
  2      1     3       3    2    0    8
         2     3       2    2    0    9
         3     6       2    1    7    0
  3      1     1       3    8    8    0
         2     5       2    5    0    7
         3     7       2    3    0    3
  4      1     3       6    7    0    7
         2     7       6    6    9    0
         3    10       5    6    0    4
  5      1     5       4    6    0    6
         2     5       4    6    6    0
         3     8       4    4    5    0
  6      1     6       7    5    0    7
         2    10       4    3    0    5
         3    10       5    1    0    6
  7      1     3       5    4    7    0
         2     3       3    5    0    6
         3     9       3    2    8    0
  8      1     5       6    6    4    0
         2     5       9    6    0    6
         3     7       4    6    6    0
  9      1     3      10    7    6    0
         2     3       9    5    0    3
         3     4       7    1    0    3
 10      1     3       7    7    3    0
         2     3       7    6    0    5
         3     9       6    5    0    5
 11      1     1       7    4    5    0
         2     7       5    4    0    3
         3     9       4    3    2    0
 12      1     2       7    7    8    0
         2     2       8    9    0    7
         3     2       8    3    8    0
 13      1     5       3    8    3    0
         2     7       2    8    0    3
         3     8       2    7    2    0
 14      1     3       5    7    5    0
         2     4       4    3    0    7
         3     4       2    3    2    0
 15      1     1       6    4    5    0
         2     6       6    4    0    4
         3     7       6    4    0    2
 16      1     6       8    8    3    0
         2     8       6    7    0    8
         3    10       4    7    3    0
 17      1     2       8    3    7    0
         2     8       6    3    7    0
         3     9       5    2    0    1
 18      1     2       9    6    0    9
         2     2       9    6    4    0
         3     9       8    4    0    9
 19      1     7       2    9    0    9
         2     7       2    9    9    0
         3     9       2    6    0    7
 20      1     5       8    8    0    9
         2     6       5    7    0    6
         3     7       4    5    2    0
 21      1     4       4    5    0    7
         2     6       3    4   10    0
         3     9       2    4    7    0
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   15   13   57   64
************************************************************************
DEADLINES:
jobnr.  deadline
  2       70
  6       17
  7       121
  8       46
  10      66
  15      7
  16      119
  17      117
  19      19
  21      124
************************************************************************
