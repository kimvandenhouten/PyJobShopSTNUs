************************************************************************
file with basedata            : md338_.bas
initial value random generator: 2003344938
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  155
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       23        6       23
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2          12  15
   3        3          3           5   6   7
   4        3          3           7   9  10
   5        3          2           8  11
   6        3          2           8  10
   7        3          3          14  16  18
   8        3          3          14  16  20
   9        3          1          15
  10        3          1          13
  11        3          2          17  18
  12        3          2          14  19
  13        3          2          16  18
  14        3          1          21
  15        3          2          17  19
  16        3          2          19  21
  17        3          1          20
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
  2      1     1       0    9    0    9
         2     2       1    0    0    3
         3     7       0    5    5    0
  3      1     7      10    0    0    6
         2     7       9    0    0    8
         3     8       3    0    0    1
  4      1     3       8    0    0    5
         2     4       0    7    0    5
         3    10       0    6    0    5
  5      1     2       7    0    0    4
         2     7       6    0    0    3
         3     9       4    0    0    3
  6      1     3       2    0    3    0
         2     9       0    3    0    8
         3     9       1    0    0    7
  7      1     8       6    0    0    7
         2    10       5    0    0    6
         3    10       0    6    7    0
  8      1     3       0    8    7    0
         2     4       0    7    5    0
         3     6       6    0    5    0
  9      1     1       5    0    0    8
         2     4       0    8    5    0
         3     6       0    6    3    0
 10      1     3       0    4    0    8
         2     7       0    3    2    0
         3    10       0    1    0    7
 11      1     6       0    5    8    0
         2     7       3    0    0    6
         3     8       3    0    0    4
 12      1     4       0    5   10    0
         2     5       7    0    0    2
         3     7       5    0    0    2
 13      1     1      10    0    0    7
         2     2      10    0    6    0
         3    10       9    0    0    6
 14      1     1       5    0    0    8
         2     6       0    4    5    0
         3     8       4    0    3    0
 15      1     1       0    9    0   10
         2     2       0    8    2    0
         3     3       0    6    0    6
 16      1     4       0    6    7    0
         2     8       0    1    2    0
         3     8      10    0    3    0
 17      1     6       0    9    0    8
         2     6       5    0    0    7
         3     7       0    6    0    5
 18      1     1       8    0    3    0
         2     7       5    0    0    8
         3     8       0    8    0    8
 19      1     3       0    2    4    0
         2     3       7    0    5    0
         3     4       3    0    0    8
 20      1     1       0    6    0    7
         2     7       0    4    0    6
         3    10       0    3    0    6
 21      1     4       8    0    7    0
         2     5       0    3    0    6
         3     7       7    0    2    0
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   19   17   63  100
************************************************************************
DEADLINES:
jobnr.  deadline
  3       46
  4       140
  5       16
  6       147
  8       49
  9       114
  10      29
  11      32
  12      121
  15      1
  16      49
  17      43
  19      99
************************************************************************
