************************************************************************
file with basedata            : md348_.bas
initial value random generator: 1703322249
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  145
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       25       15       25
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5  16  17
   3        3          3           6   8  21
   4        3          3           5   8  17
   5        3          3           7  14  21
   6        3          2           9  10
   7        3          2          10  12
   8        3          3          10  14  18
   9        3          1          11
  10        3          2          19  20
  11        3          2          12  13
  12        3          1          18
  13        3          3          14  15  16
  14        3          1          20
  15        3          1          17
  16        3          2          18  20
  17        3          1          19
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
  2      1     4       0    6    0    9
         2     7       9    0    0    7
         3     8       0    5    1    0
  3      1     1       0    1    6    0
         2     1       5    0    4    0
         3     2       0    1    1    0
  4      1     4       6    0    6    0
         2     7       4    0    0    4
         3     9       0    2    0    2
  5      1     4       2    0    8    0
         2    10       0    2    0    8
         3    10       2    0    0    6
  6      1     6       0    7    0    7
         2     8       1    0    7    0
         3    10       0    5    4    0
  7      1     1       8    0    1    0
         2     4       5    0    0    1
         3     6       0    1    0    1
  8      1     2      10    0    0    8
         2     3       0    7    5    0
         3     5       9    0    0    6
  9      1     1       7    0    2    0
         2     3       7    0    0   10
         3     3       7    0    1    0
 10      1     3       2    0    8    0
         2     3       0    6    3    0
         3     3       5    0    6    0
 11      1     1      10    0    6    0
         2     2       0    8    0    4
         3     7       0    8    0    1
 12      1    10       9    0    5    0
         2    10       0    7    5    0
         3    10       7    0    0    7
 13      1     2       6    0    7    0
         2     4       0    5    7    0
         3     5       0    4    5    0
 14      1     1       7    0    0    6
         2     1       7    0    6    0
         3     9       0    3    0    6
 15      1     7       0    6    1    0
         2     8       1    0    0    4
         3     9       0    5    0    4
 16      1     1      10    0    0    2
         2     3       0    9    0    2
         3     9       0    7    0    1
 17      1     6      10    0    3    0
         2     7       6    0    3    0
         3     8       0    8    0    3
 18      1     3       0    4   10    0
         2     5      10    0    9    0
         3     6       0    2    9    0
 19      1     1       6    0    0    7
         2     3       5    0    0    5
         3     8       4    0    6    0
 20      1     6       0    8    0    3
         2     7       4    0    7    0
         3     8       2    0    6    0
 21      1     2       3    0    0    5
         2     5       0    5    0    1
         3    10       2    0    6    0
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   32   24  101   88
************************************************************************
DEADLINES:
jobnr.  deadline
  3       1
  4       37
  10      91
  14      29
  15      121
  17      35
  20      31
  21      37
************************************************************************
