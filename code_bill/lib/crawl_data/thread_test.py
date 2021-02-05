from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

def square(n):
    return n * n

def main():
    with ThreadPoolExecutor(max_workers = 3) as executor:
        values = list(range(100))
        results = executor.map(square, values)
    for result in results:
        print(result)

if __name__ == '__main__':
    main()