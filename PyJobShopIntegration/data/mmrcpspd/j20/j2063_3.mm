************************************************************************
file with basedata            : md383_.bas
initial value random generator: 499616283
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
    1     20      0       30        9       30
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2          11  17
   3        3          3           5   7  17
   4        3          3           5   6  10
   5        3          3           8   9  14
   6        3          2          17  18
   7        3          3          12  13  15
   8        3          1          15
   9        3          1          18
  10        3          2          11  19
  11        3          3          13  14  15
  12        3          3          19  20  21
  13        3          1          16
  14        3          2          16  21
  15        3          1          16
  16        3          1          18
  17        3          2          20  21
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
  2      1     4       8    8    7    5
         2     5       7    7    6    3
         3     6       5    5    5    2
  3      1     2      10    8    9   10
         2     4       9    5    7    9
         3     6       7    5    4    8
  4      1     2       8    8    3    6
         2     7       8    8    2    3
         3     7       7    7    3    5
  5      1     3       2    7   10    5
         2     8       2    5    7    5
         3     9       2    1    7    5
  6      1     1       9    6    7    6
         2     3       7    6    5    4
         3     7       6    6    4    2
  7      1     4       4    5    5    9
         2     5       3    5    4    6
         3     5       1    5    4    8
  8      1     2      10    8    6   10
         2     5       6    6    5    9
         3    10       5    5    2    7
  9      1     1       9    5    8    6
         2     5       5    3    6    5
         3     8       4    2    5    5
 10      1     8       7   10    9    3
         2     8       9    9    6    3
         3     9       7    7    6    2
 11      1     1       6    5    6    4
         2     7       2    3    1    2
         3     7       4    2    3    3
 12      1     2       7    5    9    8
         2     4       6    4    8    7
         3    10       6    3    6    6
 13      1     8       8    7    4    5
         2     8       8    8    4    4
         3     8       3    5    5    7
 14      1     3       5    3   10    7
         2     6       5    1   10    5
         3     6       5    1    9    7
 15      1     3       7    5    5    3
         2     4       7    4    5    3
         3     4       6    5    5    3
 16      1     3       9    4    7   10
         2     3      10    4    6   10
         3     7       7    4    5    9
 17      1     7       7    6    4    9
         2     8       6    5    3    9
         3     9       6    5    3    8
 18      1     2       8    7    5    5
         2     2       9   10    4    5
         3     8       7    6    3    1
 19      1     8       9    4    7    6
         2     8       9    5    6    6
         3    10       8    2    5    6
 20      1     6       9   10    8    7
         2     8       9    8    8    6
         3    10       8    6    8    3
 21      1     5       9    4    8    7
         2     6       9    4    7    6
         3     7       8    4    1    1
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   31   29  138  133
************************************************************************
DEADLINES:
jobnr.  deadline
  3       151
  6       79
  7       52
  11      144
  13      23
************************************************************************
