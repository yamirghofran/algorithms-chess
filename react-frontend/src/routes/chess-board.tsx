import React, { useState, useEffect } from 'react'
import { QueryClient, QueryClientProvider, useQuery, useMutation } from '@tanstack/react-query'
import axios from 'axios'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'

import WhitePawnIcon from "../../public/pieces/plt.svg?react"
import BlackPawnIcon from "../../public/pieces/pdtr.svg?react"
import WhiteKnightIcon from "../../public/pieces/nlt.svg?react"
import BlackKnightIcon from "../../public/pieces/ndtr.svg?react"
import WhiteBishopIcon from "../../public/pieces/blt.svg?react"
import BlackBishopIcon from "../../public/pieces/bdtr.svg?react"
import WhiteQueenIcon from "../../public/pieces/qlt.svg?react"
import BlackQueenIcon from "../../public/pieces/qdtr.svg?react"
import WhiteKingIcon from "../../public/pieces/klt.svg?react"
import BlackKingIcon from "../../public/pieces/kdtr.svg?react"
import WhiteRookIcon from "../../public/pieces/rlt.svg?react"
import BlackRookIcon from "../../public/pieces/rdtr.svg?react"
import { Leaderboard } from '@/components/Leaderboard'

type PieceType = 
  | "white_rook" | "white_knight" | "white_bishop" | "white_queen" | "white_king" | "white_pawn"
  | "black_rook" | "black_knight" | "black_bishop" | "black_queen" | "black_king" | "black_pawn"
  | ""

type BoardState = {
  [key: string]: PieceType
}

type MovePayload = {
  start: [number, number]
  end: [number, number]
  promotion?: string
}

type GameState = {
  board: BoardState;
  currentPlayer: 'white' | 'black';
  capturedPieces: CapturedPieces;
  inCheck: boolean;
  inCheckmate: boolean;
  inStalemate: boolean;
  isDraw: boolean;
}

type CapturedPieces = {
  white: PieceType[];
  black: PieceType[];
}

const queryClient = new QueryClient()

const ChessPiece: React.FC<{ piece: PieceType; onDragStart: () => void }> = ({ piece, onDragStart }) => {
  const getPieceIcon = (piece: PieceType) => {
    switch (piece) {
      case "white_rook":
        return <WhiteRookIcon />
      case "black_rook":
        return <BlackRookIcon />
      case "white_knight":
        return <WhiteKnightIcon />
      case "black_knight":
        return <BlackKnightIcon />
      case "white_bishop":
        return <WhiteBishopIcon />
      case "black_bishop":
        return <BlackBishopIcon />
      case "white_queen":
        return <WhiteQueenIcon />
      case "black_queen":
        return <BlackQueenIcon />
      case "white_king":
        return <WhiteKingIcon />
      case "black_king":
        return <BlackKingIcon />
      case "white_pawn":
        return <WhitePawnIcon />
      case "black_pawn":
        return <BlackPawnIcon />
      default:
        return null
    }
  }

  return (
    <div
      className={`w-full h-full flex items-center justify-center ${
        piece.startsWith('white') ? 'text-white' : 'text-gray-900'
      } cursor-grab active:cursor-grabbing`}
      draggable
      onDragStart={onDragStart}
    >
      {getPieceIcon(piece)}
    </div>
  )
}

function ChessBoard() {
  const [draggedPiece, setDraggedPiece] = useState<string | null>(null)
  const [lastMove, setLastMove] = useState<{ from: string; to: string } | null>(null)
  const [promotionDialogOpen, setPromotionDialogOpen] = useState(false)
  const [promotionResolver, setPromotionResolver] = useState<((value: string | undefined) => void) | null>(null)
  const [playerName, setPlayerName] = useState<string>('')
  const [showNameDialog, setShowNameDialog] = useState(true)
  const { data: gameState, isLoading, isError, error } = useQuery<GameState>({
    queryKey: ['gameState'],
    queryFn: async () => {
      const response = await axios.get('/api/board')
      return response.data
    },
  })

  const playerMoveMutation = useMutation({
    mutationFn: (movePayload: MovePayload) => {
      return axios.post('/api/move/player', movePayload)
    },
    onSuccess: (response) => {
      if (response.data.success) {
        // Update game state with player's move
        queryClient.setQueryData(['gameState'], response.data.gameState)
        
        // Trigger CPU move
        cpuMoveMutation.mutate()
      } else {
        setLastMove(null)
        console.error('Player move was not valid:', response.data.message)
      }
    },
    onError: (error) => {
      console.error('Error making move:', error)
      setLastMove(null)
    }
  })

  const cpuMoveMutation = useMutation({
    mutationFn: () => {
      return axios.post('/api/move/cpu')
    },
    onSuccess: (response) => {
      if (response.data.success) {
        // Find the CPU's move by comparing the board states
        const previousState = queryClient.getQueryData<GameState>(['gameState'])
        const newState = response.data.gameState
        
        if (previousState && newState) {
          // Find what changed in the board
          const from = Object.entries(previousState.board).find(([pos, piece]) => 
            piece && !newState.board[pos]
          )?.[0]
          
          const to = Object.entries(newState.board).find(([pos, piece]) => 
            piece && !previousState.board[pos]
          )?.[0]

          if (from && to) {
            setLastMove({ from, to })
          }
        }

        queryClient.setQueryData(['gameState'], response.data.gameState)
      }
    }
  })

  const resetMutation = useMutation({
    mutationFn: () => axios.post('/api/reset'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['gameState'] })
      setLastMove(null)
    },
  })

  const updateLeaderboard = useMutation({
    mutationFn: async (result: 'win' | 'loss' | 'draw') => {
      return axios.post('/api/leaderboard/update', {
        name: playerName,
        result
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['leaderboard'] })
    }
  })

  const handleDragStart = (position: string) => {
    const piece = gameState.board[position];
    if (piece && piece.startsWith(gameState.currentPlayer)) {
      setDraggedPiece(position);
    } else {
      // Prevent dragging if it's not the current player's piece
      return false;
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  const handleDrop = async (targetPosition: string) => {
    if (!draggedPiece || !gameState.board) return

    const [startRow, startCol] = draggedPiece.split(',').map(Number)
    const [endRow, endCol] = targetPosition.split(',').map(Number)

    const piece = gameState.board[draggedPiece]
    const isPawnPromotion = (piece === 'white_pawn' && endRow === 7) || (piece === 'black_pawn' && endRow === 0)

    let promotionPiece: string | undefined

    if (isPawnPromotion) {
      promotionPiece = await new Promise<string | undefined>((resolve) => {
        setPromotionDialogOpen(true)
        setPromotionResolver(() => resolve)
      })

      if (!promotionPiece) {
        return // Cancel the move if no promotion piece was selected
      }
    }

    const movePayload: MovePayload = {
      start: [startRow, startCol],
      end: [endRow, endCol],
      promotion: promotionPiece
    }

    setLastMove({ from: draggedPiece, to: targetPosition })
    setDraggedPiece(null)
    playerMoveMutation.mutate(movePayload)
  }

  const handleResetBoard = () => {
    resetMutation.mutate()
  }

  const renderCapturedPieces = (color: 'white' | 'black') => (
    <div className={`flex flex-wrap gap-1  ${color === 'white' ? 'mt-4' : 'mb-4'}`}>
      {gameState?.capturedPieces[color].map((piece, index) => (
        <div key={index} className="w-8 h-8">
          {(() => {
            switch (piece) {
              case "white_rook":
                return <WhiteRookIcon />
              case "black_rook":
                return <BlackRookIcon />
              case "white_knight":
                return <WhiteKnightIcon />
              case "black_knight":
                return <BlackKnightIcon />
              case "white_bishop":
                return <WhiteBishopIcon />
              case "black_bishop":
                return <BlackBishopIcon />
              case "white_queen":
                return <WhiteQueenIcon />
              case "black_queen":
                return <BlackQueenIcon />
              case "white_king":
                return <WhiteKingIcon />
              case "black_king":
                return <BlackKingIcon />
              case "white_pawn":
                return <WhitePawnIcon />
              case "black_pawn":
                return <BlackPawnIcon />
              default:
                return null
            }
          })()}
        </div>
      ))}
    </div>
  );

  // Add this to handle game end conditions
  useEffect(() => {
    if (gameState && playerName && !showNameDialog) {
      if (gameState.inCheckmate) {
        updateLeaderboard.mutate(gameState.currentPlayer === 'white' ? 'loss' : 'win')
      } else if (gameState.inStalemate || gameState.isDraw) {
        updateLeaderboard.mutate('draw')
      }
    }
  }, [gameState?.inCheckmate, gameState?.inStalemate, gameState?.isDraw])

  if (isLoading) {
    return <div className="flex items-center justify-center min-h-screen">Loading...</div>
  }

  if (isError) {
    return <div className="flex items-center justify-center min-h-screen">Error: {error.message}</div>
  }

  if (!gameState || !gameState.board) {
    return <div className="flex items-center justify-center min-h-screen">No game state available</div>
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <Dialog open={showNameDialog} onOpenChange={setShowNameDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Enter Your Name</DialogTitle>
            <DialogDescription>
              Please enter your name to start playing and track your score.
            </DialogDescription>
          </DialogHeader>
          <Input
            value={playerName}
            onChange={(e) => setPlayerName(e.target.value)}
            placeholder="Your name"
          />
          <Button onClick={() => setShowNameDialog(false)} disabled={!playerName}>
            Start Playing
          </Button>
        </DialogContent>
      </Dialog>
      
      <Dialog open={promotionDialogOpen} onOpenChange={setPromotionDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Promote your pawn</DialogTitle>
            <DialogDescription>
              Choose a piece to promote your pawn:
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-around mt-4">
            {['queen', 'rook', 'bishop', 'knight'].map(piece => (
              <Button 
                variant="outline"
                className='p-6'
                key={piece} 
                onClick={() => {
                  console.log('Promotion piece selected:', piece)
                  promotionResolver?.(piece)
                  setPromotionDialogOpen(false)
                }}
              >
                {gameState.currentPlayer === 'white' ? (
                  piece === 'queen' ? <WhiteQueenIcon /> :
                  piece === 'rook' ? <WhiteRookIcon /> :
                  piece === 'bishop' ? <WhiteBishopIcon /> :
                  <WhiteKnightIcon />
                ) : (
                  piece === 'queen' ? <BlackQueenIcon /> :
                  piece === 'rook' ? <BlackRookIcon /> :
                  piece === 'bishop' ? <BlackBishopIcon /> :
                  <BlackKnightIcon />
                )}
              </Button>
            ))}
          </div>
          {/*<Button 
            className="mt-4 w-full" 
            onClick={() => {
              console.log('Promotion cancelled') // Add this log
              promotionResolver?.(undefined)
              setPromotionDialogOpen(false)
            }}
          >
            Cancel
          </Button>*/}
        </DialogContent>
      </Dialog>
      {renderCapturedPieces('black')}
      <div className="w-full max-w-2xl aspect-square">
        <div className="grid grid-cols-8 gap-0 border-4 border-gray-800">
          {Array.from({ length: 8 }, (_, row) =>
            Array.from({ length: 8 }, (_, col) => {
              const position = `${7 - row},${col}`
              const piece = gameState.board[position] || ""
              const isCurrentPlayerPiece = piece.startsWith(gameState.currentPlayer)
              const isLastMoveFrom = lastMove?.from === position
              const isLastMoveTo = lastMove?.to === position
              const isKingInCheck = gameState.inCheck && piece === `${gameState.currentPlayer}_king`
              const isKingInCheckmate = gameState.inCheckmate && piece === `${gameState.currentPlayer}_king`
              return (
                <div
                  key={position}
                  className={`aspect-square flex items-center justify-center ${
                    ((7 - row) + col) % 2 === 0 ? 'bg-amber-200' : 'bg-amber-800'
                  } ${draggedPiece && isCurrentPlayerPiece ? 'hover:bg-green-200' : ''}
                    ${isLastMoveFrom ? 'bg-green-300' : ''}
                    ${isLastMoveTo ? 'bg-green-500' : ''}
                    ${isKingInCheck ? 'bg-red-500' : ''}
                    ${isKingInCheckmate ? 'bg-gray-500' : ''}
                    ${isCurrentPlayerPiece ? 'cursor-grab' : 'cursor-not-allowed'}`}
                  onDragOver={handleDragOver}
                  onDrop={() => handleDrop(position)}
                  tabIndex={0}
                >
                  {piece && (
                    <ChessPiece
                      piece={piece}
                      onDragStart={() => handleDragStart(position)}
                    />
                  )}
                </div>
              )
            })
          )}
        </div>
      </div>
      {renderCapturedPieces('white')}
      <div className="mt-4 text-lg font-bold">
        {gameState.inCheckmate ? (
          <span className="text-red-600">Checkmate! {gameState.currentPlayer === 'white' ? 'Black' : 'White'} wins!</span>
        ) : gameState.inStalemate ? (
          <span className="text-yellow-600">Stalemate! The game is a draw.</span>
        ) : gameState.isDraw ? (
          <span className="text-yellow-600">Draw! The game has ended in a draw.</span>
        ) : (
          <>
            Current Player: {gameState.currentPlayer}
            {gameState.inCheck && <span className="text-red-600"> (In Check)</span>}
          </>
        )}
      </div>
      <Button 
        onClick={handleResetBoard}
        disabled={resetMutation.isPending}
        className="mt-4"
      >
        {resetMutation.isPending ? 'Resetting...' : 'Reset Board'}
      </Button>
      <Leaderboard />
    </div>
  )
}

export default function Component() {
  return (
    <QueryClientProvider client={queryClient}>
      <ChessBoard />
    </QueryClientProvider>
  )
}
