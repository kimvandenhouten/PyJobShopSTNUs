************************************************************************
file with basedata            : md326_.bas
initial value random generator: 1054013861
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  151
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       21        0       21
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   7  13
   3        3          2           6  14
   4        3          3           5   6  17
   5        3          2          11  18
   6        3          1          12
   7        3          3           8   9  10
   8        3          1          15
   9        3          1          16
  10        3          3          11  15  16
  11        3          2          12  21
  12        3          2          19  20
  13        3          2          16  20
  14        3          3          15  17  20
  15        3          2          19  21
  16        3          2          17  21
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
  2      1     1      10    6    0    9
         2     4       9    5    3    0
         3     5       9    5    1    0
  3      1     7       3    7    0    5
         2     8       2    5    9    0
         3     9       2    5    5    0
  4      1     7       6   10    9    0
         2    10       5    6    3    0
         3    10       4    8    0    9
  5      1     2       3    7    8    0
         2     9       3    6    5    0
         3     9       2    4    6    0
  6      1     3       9    8   10    0
         2     4       8    6    6    0
         3     4       8    6    0    7
  7      1     3       3    4    1    0
         2     3       3    4    0    7
         3     8       1    3    0    6
  8      1     2       6    6    6    0
         2     4       4    5    0    7
         3     7       2    3    1    0
  9      1     6      10    4    0    3
         2     8      10    3    0    2
         3    10      10    2    0    2
 10      1     7       2   10    1    0
         2     8       2    7    0    7
         3    10       1    6    0    4
 11      1     2       4    7   10    0
         2     4       4    5    0   10
         3     5       2    4    0    2
 12      1     1       7    2    0    5
         2     5       7    2    9    0
         3     7       6    2    4    0
 13      1     2       3    5    4    0
         2     2       2    4    5    0
         3     9       2    1    4    0
 14      1     2       4    4    0    4
         2     4       2    3    0    4
         3     5       1    1    0    3
 15      1     3       2    3    0    6
         2     8       2    1   10    0
         3     8       1    3    0    6
 16      1     2       8    3    0    5
         2    10       5    2    4    0
         3    10       4    3    0    4
 17      1     5       6   10    5    0
         2     6       6    9    2    0
         3     9       5    8    0    5
 18      1     2      10    4    7    0
         2     3       6    3    0    1
         3     7       3    3    5    0
 19      1     1       5    5    0    9
         2     2       4    4   10    0
         3     3       4    4    0    9
 20      1     1       7    7    0    6
         2     6       4    5    0    5
         3     9       3    3    0    5
 21      1     2       5   10    0    6
         2     6       4    6    7    0
         3     7       3    4    0    3
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   19   23   35   35
************************************************************************
DEADLINES:
jobnr.  deadline
  14      131
  18      68
  21      118
************************************************************************
