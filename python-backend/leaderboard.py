class LeaderboardEntry:
    # O(1) - Simple initialization with constant time operations
    def __init__(self, name: str, wins: int = 0, losses: int = 0, draws: int = 0):
        self.name = name
        self.wins = wins
        self.losses = losses
        self.draws = draws
        self.score = self.calculate_score()
    
    # O(1) - Simple arithmetic operations
    def calculate_score(self):
        return (self.wins * 2) + (self.draws * 1) - (self.losses * 1)
    
    # O(1) - String formatting with constant time operations
    def to_csv_line(self):
        return f"{self.name},{self.wins},{self.losses},{self.draws}\n"
    
    # O(1) - String splitting and conversion operations are constant time
    @staticmethod
    def from_csv_line(line: str):
        name, wins, losses, draws = line.strip().split(',')
        return LeaderboardEntry(name, int(wins), int(losses), int(draws))

# O(n) - Single pass through array from low to high
def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j].score >= pivot.score:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# O(n log n) average case, O(n²) worst case
# Average: Divides array in half each time (log n) with n operations per level
# Worst: When already sorted or reverse sorted, partitions become unbalanced
def quick_sort_leaderboard(arr, low=None, high=None):
    if low is None:
        low = 0
    if high is None:
        high = len(arr) - 1
        
    if low < high:
        pi = partition(arr, low, high)
        quick_sort_leaderboard(arr, low, pi - 1)
        quick_sort_leaderboard(arr, pi + 1, high)
    
    return arr