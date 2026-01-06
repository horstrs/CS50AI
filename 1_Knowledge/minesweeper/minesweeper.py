import itertools
import random


class Minesweeper:
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):
        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence:
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if self.count == len(self.cells):
            return self.cells
        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return self.cells
        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell not in self.cells:
            return
        self.cells.remove(cell)
        self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell not in self.cells:
            return
        self.cells.remove(cell)


class MinesweeperAI:
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):
        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()
        self.all_moves = self.calculate_all_moves()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        self.moves_made.add(cell)
        self.mark_safe(cell)
        unknown_neighbours = self.get_unknown_neighbours(cell)
        new_knowledge = Sentence(unknown_neighbours, count)
        self.knowledge.append(new_knowledge)
        mines_in_new_knowledge = new_knowledge.cells.intersection(self.mines)
        for mine in mines_in_new_knowledge:
            self.mark_mine(mine)
        
        self.reason_about_new_knowledge(new_knowledge)

    def reason_about_new_knowledge(self, new_knowledge):
        added_knowledge = True
        if len(new_knowledge.cells) == 0:
            return
        
        while added_knowledge:
            added_knowledge = False
            self.clean_up_empty_knowledge()
            known_safes = set()
            known_mines = set()

            previous_safes_count = len(self.safes)
            previous_mines_count = len(self.mines)

            for sentence in self.knowledge:
                for mine in sentence.known_mines():
                    known_mines.add(mine)
                for safe in sentence.known_safes():
                    known_safes.add(safe)

            for mine in known_mines:
                self.mark_mine(mine)
            for safe in known_safes:
                self.mark_safe(safe)

            new_safes_count = len(self.safes)
            new_mines_count = len(self.mines)

            if (
                previous_safes_count != new_safes_count
                or previous_mines_count != new_mines_count
            ):
                added_knowledge = True
                
        for existing_knowledge in self.knowledge:
            added_knowledge = False
            if len(existing_knowledge.cells) == 0:
                self.clean_up_empty_knowledge()
                continue
            if (
                existing_knowledge.cells.issuperset(new_knowledge.cells)
                and existing_knowledge.cells != new_knowledge.cells
            ):
                existing_knowledge.cells.difference_update(new_knowledge.cells)
                existing_knowledge.count -= new_knowledge.count
                added_knowledge = True
            if added_knowledge:
                self.reason_about_new_knowledge(existing_knowledge)

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for safe in self.safes:
            if safe not in self.moves_made:
                return safe
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        remaining_moves = self.all_moves.difference(self.moves_made)
        for move in remaining_moves:
            if move in self.mines:
                continue
            return move
        return None

    def get_unknown_neighbours(self, cell):
        neighbours = set()
        row, col = cell
        for i in range(row - 1, row + 2):
            if i < 0 or i >= self.height:
                continue
            for j in range(col - 1, col + 2):
                if j < 0 or j >= self.width:
                    continue
                if (i, j) not in self.safes: #and (i, j) not in self.mines:
                    neighbours.add((i, j))
        return neighbours

    def calculate_all_moves(self):
        all_moves = set()
        for i in range(0, self.height):
            for j in range(0, self.width):
                all_moves.add((i, j))
        return all_moves

    def clean_up_empty_knowledge(self):
        sentences_to_clean = []
        for i in range(0, len(self.knowledge)):
            if len(self.knowledge[i].cells) == 0:
                sentences_to_clean.append(i)
        sentences_to_clean.reverse()
        for i in sentences_to_clean:
            self.knowledge.pop(i)
    
    def check_if_sentence_has_known_mines(self, sentence):
        known_mines_in_sentence = self.mines.intersection(sentence.cells)
        return known_mines_in_sentence