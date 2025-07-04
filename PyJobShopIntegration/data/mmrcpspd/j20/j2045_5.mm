************************************************************************
file with basedata            : md365_.bas
initial value random generator: 1892075516
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  156
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       22       17       22
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   6  10
   3        3          1           8
   4        3          2           6  10
   5        3          2           8   9
   6        3          2           7  14
   7        3          2          16  21
   8        3          3          11  12  13
   9        3          2          18  20
  10        3          3          12  16  18
  11        3          1          16
  12        3          2          14  21
  13        3          2          14  15
  14        3          1          17
  15        3          2          17  18
  16        3          2          19  20
  17        3          2          19  20
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
  2      1     4       9    9    9    8
         2     5       9    6    5    7
         3     6       8    4    4    4
  3      1     1       6    9    8    6
         2     2       5    8    7    6
         3     9       3    8    6    2
  4      1     2       3    7    9    6
         2     7       3    3    7    6
         3     7       3    5    7    1
  5      1     1       7    6    5    6
         2     2       7    5    4    6
         3     6       7    3    1    3
  6      1     8       6    9    4    5
         2     9       4    6    4    5
         3    10       2    5    3    5
  7      1     1       5    8   10    9
         2     8       4    3    5    9
         3     8       3    6    7    9
  8      1     3      10    8    6    7
         2     3      10    6    7    6
         3     5       9    4    5    3
  9      1     6       7    7    6    9
         2     8       5    6    1    8
         3     8       6    3    2    7
 10      1     8       6    8    6    7
         2    10       3    5    6    4
         3    10       2    7    6    3
 11      1     3       5    6    9    7
         2     5       5    5    7    7
         3     6       4    3    7    7
 12      1     1       3    6   10    1
         2     6       2    6    9    1
         3     9       2    5    8    1
 13      1     3       5    3    9    5
         2     8       5    2    9    4
         3     8       5    3    8    2
 14      1     2       2    9    9    6
         2     6       2    9    8    5
         3     6       1    9    9    4
 15      1     3       9    5    4    4
         2     5       6    1    4    3
         3     5       6    4    3    3
 16      1     2       7    6    6   10
         2     3       6    6    6   10
         3     8       5    3    5   10
 17      1     3       9    8    7    6
         2     6       6    7    4    3
         3     9       3    6    2    3
 18      1     1       3    7   10    7
         2     3       2    7    8    6
         3    10       2    6    7    6
 19      1     4      10    5    6    6
         2     7       9    5    6    3
         3     9       8    5    5    2
 20      1     3       9    6    8    5
         2     6       8    5    8    2
         3     8       7    4    8    1
 21      1     3       8    8    6    8
         2     3       5    8    7    8
         3     9       3    8    4    8
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   14   15  126  106
************************************************************************
DEADLINES:
jobnr.  deadline
  2       29
  4       7
  5       39
  6       23
  8       107
  11      52
  12      33
  13      117
  15      98
  17      55
  18      93
  20      143
************************************************************************
