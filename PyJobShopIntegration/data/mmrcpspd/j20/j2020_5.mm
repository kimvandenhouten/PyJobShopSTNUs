************************************************************************
file with basedata            : md340_.bas
initial value random generator: 1517989689
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  152
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       20        0       20
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           8   9  17
   3        3          2           6   9
   4        3          3           5   6  11
   5        3          2          12  21
   6        3          3           7  15  16
   7        3          1          10
   8        3          3          14  15  18
   9        3          2          16  21
  10        3          2          13  19
  11        3          1          15
  12        3          3          14  17  18
  13        3          2          17  18
  14        3          1          16
  15        3          2          19  21
  16        3          2          19  20
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
  2      1     5       0    2    0    3
         2     5       4    0    0    3
         3     9       0    2    9    0
  3      1     1       6    0    0    7
         2     4       3    0    4    0
         3     4       0    7    0    5
  4      1     3       0    4    9    0
         2     8       0    3    7    0
         3     9       4    0    5    0
  5      1     1       0    6    0    7
         2     4       0    4    6    0
         3     6       4    0    0    5
  6      1     3       8    0    4    0
         2     3       0    7    0    7
         3    10       8    0    3    0
  7      1     3       8    0    0    5
         2     4       6    0    0    4
         3     4       0    6    0    4
  8      1     3       0    8    9    0
         2     3       6    0    0    8
         3     5       5    0    9    0
  9      1     4       0    9    0    9
         2     9       0    7    7    0
         3    10       6    0    5    0
 10      1     2       0    7    8    0
         2     2       0    6    0    6
         3     4       0    3    8    0
 11      1     7       4    0    0    7
         2     7       0    3    0    7
         3     9       0    2    0    7
 12      1     7       0    7    0    7
         2    10       6    0    4    0
         3    10       0    4    4    0
 13      1     1       0    5    0   10
         2     4       0    5    0    8
         3     7       0    4    0    7
 14      1     3       0   10    0    9
         2     4       0    9    0    5
         3    10       0    8    5    0
 15      1     1       0    9    0    5
         2     3       0    8    5    0
         3     8      10    0    0    1
 16      1     2       6    0    0    6
         2     5       5    0    9    0
         3     8       3    0    6    0
 17      1     1       2    0    2    0
         2     8       1    0    0    8
         3     9       0    5    0    8
 18      1     6       0    2    0    9
         2     9      10    0    0    5
         3     9      10    0    3    0
 19      1     4       2    0    0    8
         2     5       2    0    6    0
         3     9       2    0    0    6
 20      1     2       2    0    7    0
         2     4       0    9    0    6
         3     6       0    6    5    0
 21      1     2       2    0    0    4
         2     6       1    0    0    4
         3     6       0    1    6    0
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   22   34   79  103
************************************************************************
DEADLINES:
jobnr.  deadline
  3       104
  10      39
  13      101
  16      104
  21      52
************************************************************************
