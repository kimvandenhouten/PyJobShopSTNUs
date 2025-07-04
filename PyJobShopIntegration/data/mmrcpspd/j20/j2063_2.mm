************************************************************************
file with basedata            : md383_.bas
initial value random generator: 228501733
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  167
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       25       11       25
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           8  11  15
   3        3          2           5   8
   4        3          3           6  13  21
   5        3          3           9  10  12
   6        3          2           7  10
   7        3          3          12  14  15
   8        3          2          14  21
   9        3          3          11  16  20
  10        3          2          15  17
  11        3          3          13  17  21
  12        3          1          18
  13        3          1          19
  14        3          2          16  20
  15        3          1          20
  16        3          1          17
  17        3          1          18
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
  2      1     3       9    5    4    7
         2     6       9    5    4    5
         3    10       8    4    3    4
  3      1     2       5    7    6    8
         2     2       4    9    5    8
         3     7       4    5    4    7
  4      1     2      10    9    7    6
         2     4       9    9    6    6
         3     9       9    8    5    5
  5      1     2       9    7    6    9
         2     7       5    4    5    5
         3    10       3    2    4    2
  6      1     5       5    9    6    6
         2     9       5    8    3    6
         3     9       5    9    2    6
  7      1     2       7    5    6    5
         2    10       6    4    6    4
         3    10       3    5    6    4
  8      1     7       7    8    2   10
         2     9       7    7    2    9
         3    10       6    7    1    9
  9      1     2       8    3    8    7
         2     6       7    2    5    6
         3     9       5    2    4    6
 10      1     2       3    7    6    7
         2     4       3    5    6    7
         3     5       2    2    3    6
 11      1     3       9    7    7    7
         2     5       6    6    5    4
         3     9       3    4    2    1
 12      1     1       5    3    7    5
         2     1       5    3    5    7
         3     5       2    3    3    4
 13      1     2       9   10    5    5
         2     3       8    9    3    4
         3     5       8    8    2    3
 14      1     2       3    7    6   10
         2     2       4    6    6   10
         3     9       2    6    3   10
 15      1     4       4    9    6    9
         2     8       3    9    6    8
         3     9       3    9    5    6
 16      1     5       6    4    5    5
         2     6       5    4    4    2
         3    10       4    4    3    2
 17      1     5       7    6    3    7
         2     6       5    4    2    6
         3     8       1    2    2    3
 18      1     2       1    3    2    6
         2     6       1    2    1    4
         3     6       1    2    2    3
 19      1     1       9    9    7    4
         2     2       9    5    6    4
         3     9       8    2    5    1
 20      1     6       6    8    2    5
         2     7       5    6    2    5
         3     8       1    3    1    5
 21      1     3       4    9    3    9
         2     5       4    6    2    8
         3    10       4    5    2    6
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   22   29  104  139
************************************************************************
DEADLINES:
jobnr.  deadline
  4       66
  7       112
  10      138
  11      72
  12      52
  13      77
  14      76
  15      92
************************************************************************
