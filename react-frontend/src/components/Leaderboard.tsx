import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'

interface LeaderboardEntry {
  name: string
  wins: number
  losses: number
  draws: number
  score: number
}

export function Leaderboard() {
  const { data: entries = [] } = useQuery<LeaderboardEntry[]>({
    queryKey: ['leaderboard'],
    queryFn: async () => {
      const response = await axios.get('/api/leaderboard')
      return response.data
    },
  })

  return (
    <div className="w-full max-w-2xl mx-auto mt-8">
      <h2 className="text-2xl font-bold mb-4">Leaderboard</h2>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Rank</TableHead>
            <TableHead>Name</TableHead>
            <TableHead>Wins</TableHead>
            <TableHead>Losses</TableHead>
            <TableHead>Draws</TableHead>
            <TableHead>Score</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {entries.map((entry, index) => (
            <TableRow key={entry.name}>
              <TableCell>{index + 1}</TableCell>
              <TableCell>{entry.name}</TableCell>
              <TableCell>{entry.wins}</TableCell>
              <TableCell>{entry.losses}</TableCell>
              <TableCell>{entry.draws}</TableCell>
              <TableCell>{entry.score}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
} 