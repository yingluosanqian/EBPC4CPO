#include <bits/stdc++.h>
using namespace std;



int main()
{
	int h;
    int w;
    int **board;
    int **movevalues;
    int i;
    int j;
    char hold;
    cin >> h;
    cin >> w;
    
    board = (int **)malloc(sizeof(int *) * h);
    movevalues = (int **)malloc(sizeof(int *) * h);
    i = 0;
    while (i < h)
    {
        board[i] = (int *)malloc(sizeof(int) * w);
        movevalues[i] = (int *)malloc(sizeof(int) * w);
        i++;
    }
    i = 0;
    while (i < h)
    {
        j = 0;
        while (j < w)
        {
            cin >> hold;
            if (hold == '+')
                board[i][j] = 1;
            else
                board[i][j] = -1;
            j++;
        }
        i++;
    }
    i = (h - 1);
    while (i > -1)
    {
        j = (w - 1);
        while (j > -1)
        {
            if ((i == (h - 1)) && j == (w - 1))
                movevalues[i][j] = 0;
            else if (i == (h - 1))
                movevalues[i][j] = board[i][j + 1] - movevalues[i][j + 1];
            else if (j == (w - 1))
                movevalues[i][j] = board[i + 1][j] - movevalues[i + 1][j];
            else
            {
                movevalues[i][j] = max(board[i + 1][j] - movevalues[i + 1][j],\
                                       board[i][j + 1] - movevalues[i][j + 1]);
            }
            j--;
        }
        i--;
    }
    if (movevalues[0][0] > 0)
        cout << "Takahashi" << endl;
    else if (movevalues[0][0] < 0)
        cout << "Aoki" << endl;
    else if (movevalues[0][0] == 0)
        cout << "Draw" << endl;
    i = 0;
    while (i < h)
    {
        free(board[i]);
        free(movevalues[i]);
        i++;
    }
    free(board);
    free(movevalues);
}
