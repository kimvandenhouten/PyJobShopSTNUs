************************************************************************
file with basedata            : md381_.bas
initial value random generator: 25739
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  162
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       29       10       29
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5  11  16
   3        3          3           5   7  10
   4        3          1           8
   5        3          2           6  17
   6        3          3          14  18  19
   7        3          1           9
   8        3          2          11  13
   9        3          3          12  14  16
  10        3          2          11  12
  11        3          2          14  19
  12        3          2          17  21
  13        3          2          16  18
  14        3          1          15
  15        3          2          20  21
  16        3          1          17
  17        3          2          19  20
  18        3          2          20  21
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     1       9    6    6    8
         2     3       9    5    4    8
         3     5       9    3    4    7
  3      1     3       5    4    4    7
         2     3       5    5    4    6
         3     9       5    4    4    4
  4      1     2      10    4   10    9
         2     8       9    4    9    7
         3    10       9    3    9    7
  5      1     5       8    8    5   10
         2     7       7    8    5    6
         3     9       3    6    3    2
  6      1     5       9   10    6    4
         2     6       8    9    6    4
         3     7       6    8    5    4
  7      1     3       7    7    3    8
         2     5       6    5    3    8
         3    10       4    2    2    8
  8      1     3       7    7    4    7
         2     3       8    7    3    7
         3     7       5    7    2    3
  9      1     3       7    7    7    6
         2     5       3    4    5    6
         3     5       2    2    7    6
 10      1     5       8    7    2    8
         2     6       7    6    2    8
         3     9       5    4    1    7
 11      1     5       9    8    7    5
         2     7       9    8    5    5
         3     9       9    7    5    5
 12      1     3       7    9    7    3
         2     5       7    9    6    2
         3     8       6    9    4    2
 13      1     1       4    5    7    4
         2     4       2    5    7    4
         3     9       1    4    3    3
 14      1     5       7   10   10    9
         2     8       5    9    6    8
         3     8       4    9    8    7
 15      1     9       9    5    9    8
         2     9       8    8    8    6
         3    10       7    2    6    5
 16      1     2       7    7    5    7
         2     5       7    6    5    7
         3     6       7    4    5    7
 17      1     3       6    9    3    4
         2     4       4    8    3    4
         3     6       4    8    2    1
 18      1     5       3    9   10    3
         2     8       2    5    7    2
         3     9       1    5    7    2
 19      1     2       7    3    4    8
         2     4       5    3    4    5
         3     9       3    3    4    5
 20      1     2       6    4   10    7
         2     3       6    3    8    6
         3     8       6    1    8    5
 21      1     2       3    4    7    7
         2     9       2    3    4    5
         3     9       1    3    7    5
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   15   15  126  132
************************************************************************
DEADLINES:
jobnr.  deadline
  2       40
  3       122
  4       9
  5       81
  8       93
  9       151
  10      68
  11      134
  12      8
  13      153
  14      134
  16      139
  17      153
  19      19
  20      72
  21      99
************************************************************************
