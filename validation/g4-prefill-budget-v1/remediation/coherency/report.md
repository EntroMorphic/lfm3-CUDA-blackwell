| Case | Engine | Prompt tokens | New tokens | Stopped | Text = HF decode | Repeated 4-grams | Common prefix with native | Fact recalled |
|---|---|---|---|---|---|---|---|---|
| sky-512 | native | 21 (HF 21) | 72 | yes | yes | 0.00 | 72 |  |
| sky-512 | cublas | 21 (HF 21) | 72 | yes | yes | 0.00 | 72 |  |
| sky-2560 | native | 21 (HF 21) | 72 | yes | yes | 0.00 | 72 |  |
| sky-2560 | cublas | 21 (HF 21) | 72 | yes | yes | 0.00 | 72 |  |
| sky-4096 | native | 21 (HF 21) | 72 | yes | yes | 0.00 | 72 |  |
| sky-4096 | cublas | 21 (HF 21) | 72 | yes | yes | 0.00 | 72 |  |
| sky-32768 | native | 21 (HF 21) | 72 | yes | yes | 0.00 | 72 |  |
| sky-32768 | cublas | 21 (HF 21) | 72 | yes | yes | 0.00 | 72 |  |
| code-512 | native | 30 (HF 30) | 256 | **limit** | yes | 0.04 | 256 |  |
| code-512 | cublas | 30 (HF 30) | 256 | **limit** | yes | 0.04 | 84 |  |
| code-2560 | native | 30 (HF 30) | 256 | **limit** | yes | 0.04 | 256 |  |
| code-2560 | cublas | 30 (HF 30) | 256 | **limit** | yes | 0.04 | 84 |  |
| code-4096 | native | 30 (HF 30) | 256 | **limit** | yes | 0.04 | 256 |  |
| code-4096 | cublas | 30 (HF 30) | 256 | **limit** | yes | 0.04 | 84 |  |
| code-32768 | native | 30 (HF 30) | 256 | **limit** | yes | 0.04 | 256 |  |
| code-32768 | cublas | 30 (HF 30) | 256 | **limit** | yes | 0.04 | 84 |  |
| arith-512 | native | 41 (HF 41) | 121 | yes | yes | 0.14 | 121 |  |
| arith-512 | cublas | 41 (HF 41) | 121 | yes | yes | 0.14 | 121 |  |
| arith-2560 | native | 41 (HF 41) | 121 | yes | yes | 0.14 | 121 |  |
| arith-2560 | cublas | 41 (HF 41) | 121 | yes | yes | 0.14 | 121 |  |
| arith-4096 | native | 41 (HF 41) | 121 | yes | yes | 0.14 | 121 |  |
| arith-4096 | cublas | 41 (HF 41) | 121 | yes | yes | 0.14 | 121 |  |
| arith-32768 | native | 41 (HF 41) | 121 | yes | yes | 0.14 | 121 |  |
| arith-32768 | cublas | 41 (HF 41) | 121 | yes | yes | 0.14 | 121 |  |
| poem-512 | native | 22 (HF 22) | 41 | yes | yes | 0.00 | 41 |  |
| poem-512 | cublas | 22 (HF 22) | 41 | yes | yes | 0.00 | 41 |  |
| poem-2560 | native | 22 (HF 22) | 41 | yes | yes | 0.00 | 41 |  |
| poem-2560 | cublas | 22 (HF 22) | 41 | yes | yes | 0.00 | 41 |  |
| poem-4096 | native | 22 (HF 22) | 41 | yes | yes | 0.00 | 41 |  |
| poem-4096 | cublas | 22 (HF 22) | 41 | yes | yes | 0.00 | 41 |  |
| poem-32768 | native | 22 (HF 22) | 41 | yes | yes | 0.00 | 41 |  |
| poem-32768 | cublas | 22 (HF 22) | 41 | yes | yes | 0.00 | 41 |  |
| memory-512 | native | 64 (HF 64) | 36 | yes | yes | 0.00 | 36 |  |
| memory-512 | cublas | 64 (HF 64) | 40 | yes | yes | 0.00 | 19 |  |
| memory-2560 | native | 64 (HF 64) | 36 | yes | yes | 0.00 | 36 |  |
| memory-2560 | cublas | 64 (HF 64) | 40 | yes | yes | 0.00 | 19 |  |
| memory-4096 | native | 64 (HF 64) | 36 | yes | yes | 0.00 | 36 |  |
| memory-4096 | cublas | 64 (HF 64) | 40 | yes | yes | 0.00 | 19 |  |
| memory-32768 | native | 64 (HF 64) | 36 | yes | yes | 0.00 | 36 |  |
| memory-32768 | cublas | 64 (HF 64) | 40 | yes | yes | 0.00 | 19 |  |
| doc-2560 | native | 2055 (HF 2055) | 51 | yes | yes | 0.00 | 51 | yes |
| doc-2560 | cublas | 2055 (HF 2055) | 51 | yes | yes | 0.00 | 51 | yes |
| doc-4096 | native | 3555 (HF 3555) | 93 | yes | yes | 0.00 | 93 | yes |
| doc-4096 | cublas | 3555 (HF 3555) | 83 | yes | yes | 0.00 | 24 | yes |
| doc-32768 | native | 30054 (HF 30054) | 155 | yes | yes | 0.00 | 155 | yes |
| doc-32768 | cublas | 30054 (HF 30054) | 156 | yes | yes | 0.00 | 9 | yes |

### sky-512

**native:**

```text
The sky appears blue because sunlight contains all colors, but the atmosphere scatters the shorter blue wavelengths more efficiently than the longer red ones. This scattering, known as Rayleigh scattering, redirects blue light in all directions, filling the sky with a blue glow. As a result, we see blue from every direction, even though the sun itself is white.
```

**cublas:**

```text
The sky appears blue because sunlight contains all colors, but the atmosphere scatters the shorter blue wavelengths more efficiently than the longer red ones. This scattering, known as Rayleigh scattering, redirects blue light in all directions, filling the sky with a blue glow. As a result, we see blue from every direction, even though the sun itself is white.
```


### sky-2560

**native:**

```text
The sky appears blue because sunlight contains all colors, but the atmosphere scatters the shorter blue wavelengths more efficiently than the longer red ones. This scattering, known as Rayleigh scattering, redirects blue light in all directions, filling the sky with a blue glow. As a result, we see blue from every direction, even though the sun itself is white.
```

**cublas:**

```text
The sky appears blue because sunlight contains all colors, but the atmosphere scatters the shorter blue wavelengths more efficiently than the longer red ones. This scattering, known as Rayleigh scattering, redirects blue light in all directions, filling the sky with a blue glow. As a result, we see blue from every direction, even though the sun itself is white.
```


### sky-4096

**native:**

```text
The sky appears blue because sunlight contains all colors, but the atmosphere scatters the shorter blue wavelengths more efficiently than the longer red ones. This scattering, known as Rayleigh scattering, redirects blue light in all directions, filling the sky with a blue glow. As a result, we see blue from every direction, even though the sun itself is white.
```

**cublas:**

```text
The sky appears blue because sunlight contains all colors, but the atmosphere scatters the shorter blue wavelengths more efficiently than the longer red ones. This scattering, known as Rayleigh scattering, redirects blue light in all directions, filling the sky with a blue glow. As a result, we see blue from every direction, even though the sun itself is white.
```


### sky-32768

**native:**

```text
The sky appears blue because sunlight contains all colors, but the atmosphere scatters the shorter blue wavelengths more efficiently than the longer red ones. This scattering, known as Rayleigh scattering, redirects blue light in all directions, filling the sky with a blue glow. As a result, we see blue from every direction, even though the sun itself is white.
```

**cublas:**

```text
The sky appears blue because sunlight contains all colors, but the atmosphere scatters the shorter blue wavelengths more efficiently than the longer red ones. This scattering, known as Rayleigh scattering, redirects blue light in all directions, filling the sky with a blue glow. As a result, we see blue from every direction, even though the sun itself is white.
```


### code-512

**native:**

```text
To solve this problem, we need to write a function that returns the n-th Fibonacci number iteratively. The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, starting from 0 and 1. The function should efficiently compute the n-th Fibonacci number using an iterative approach.

### Approach
The approach to solve this problem involves using an iterative method to compute the Fibonacci sequence. Here's a step-by-step breakdown of the approach:

1. **Base Cases**: If n is 0, return 0. If n is 1, return 1. These are the starting points of the Fibonacci sequence.
2. **Iterative Calculation**: For values of n greater than 1, use a loop to compute each Fibonacci number up to the n-th number. We maintain two variables, `a` and `b`, to keep track of the two most recent Fibonacci numbers. In each iteration, compute the next Fibonacci number as the sum of `a` and `b`, then update `a` and `b` to the next pair of numbers.
3. **Efficiency**: This approach runs in O(n) time complexity because it iterates from 2 to n, performing a constant amount of work in each iteration. The space
```

**cublas:**

```text
To solve this problem, we need to write a function that returns the n-th Fibonacci number iteratively. The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, starting from 0 and 1. The function should efficiently compute the n-th Fibonacci number using an iterative approach.

### Approach
The approach to solve this problem involves using an iterative method to compute the Fibonacci sequence. Here are the key steps:

1. **Base Cases**: If n is 0, return 0. If n is 1, return 1. These are the starting points of the Fibonacci sequence.
2. **Iterative Calculation**: For values of n greater than 1, use a loop to compute each Fibonacci number up to the n-th number. We maintain two variables, `a` and `b`, to keep track of the two most recent Fibonacci numbers. In each iteration, compute the next Fibonacci number as the sum of `a` and `b`, then update `a` and `b` to the next pair of numbers.
3. **Efficiency**: This approach runs in O(n) time complexity because each iteration from 2 to n performs a constant amount of work. The space complexity is O(1) since we only
```


### code-2560

**native:**

```text
To solve this problem, we need to write a function that returns the n-th Fibonacci number iteratively. The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, starting from 0 and 1. The function should efficiently compute the n-th Fibonacci number using an iterative approach.

### Approach
The approach to solve this problem involves using an iterative method to compute the Fibonacci sequence. Here's a step-by-step breakdown of the approach:

1. **Base Cases**: If n is 0, return 0. If n is 1, return 1. These are the starting points of the Fibonacci sequence.
2. **Iterative Calculation**: For values of n greater than 1, use a loop to compute each Fibonacci number up to the n-th number. We maintain two variables, `a` and `b`, to keep track of the two most recent Fibonacci numbers. In each iteration, compute the next Fibonacci number as the sum of `a` and `b`, then update `a` and `b` to the next pair of numbers.
3. **Efficiency**: This approach runs in O(n) time complexity because it iterates from 2 to n, performing a constant amount of work in each iteration. The space
```

**cublas:**

```text
To solve this problem, we need to write a function that returns the n-th Fibonacci number iteratively. The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, starting from 0 and 1. The function should efficiently compute the n-th Fibonacci number using an iterative approach.

### Approach
The approach to solve this problem involves using an iterative method to compute the Fibonacci sequence. Here are the key steps:

1. **Base Cases**: If n is 0, return 0. If n is 1, return 1. These are the starting points of the Fibonacci sequence.
2. **Iterative Calculation**: For values of n greater than 1, use a loop to compute each Fibonacci number up to the n-th number. We maintain two variables, `a` and `b`, to keep track of the two most recent Fibonacci numbers. In each iteration, compute the next Fibonacci number as the sum of `a` and `b`, then update `a` and `b` to the next pair of numbers.
3. **Efficiency**: This approach runs in O(n) time complexity because each iteration from 2 to n performs a constant amount of work. The space complexity is O(1) since we only
```


### code-4096

**native:**

```text
To solve this problem, we need to write a function that returns the n-th Fibonacci number iteratively. The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, starting from 0 and 1. The function should efficiently compute the n-th Fibonacci number using an iterative approach.

### Approach
The approach to solve this problem involves using an iterative method to compute the Fibonacci sequence. Here's a step-by-step breakdown of the approach:

1. **Base Cases**: If n is 0, return 0. If n is 1, return 1. These are the starting points of the Fibonacci sequence.
2. **Iterative Calculation**: For values of n greater than 1, use a loop to compute each Fibonacci number up to the n-th number. We maintain two variables, `a` and `b`, to keep track of the two most recent Fibonacci numbers. In each iteration, compute the next Fibonacci number as the sum of `a` and `b`, then update `a` and `b` to the next pair of numbers.
3. **Efficiency**: This approach runs in O(n) time complexity because it iterates from 2 to n, performing a constant amount of work in each iteration. The space
```

**cublas:**

```text
To solve this problem, we need to write a function that returns the n-th Fibonacci number iteratively. The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, starting from 0 and 1. The function should efficiently compute the n-th Fibonacci number using an iterative approach.

### Approach
The approach to solve this problem involves using an iterative method to compute the Fibonacci sequence. Here are the key steps:

1. **Base Cases**: If n is 0, return 0. If n is 1, return 1. These are the starting points of the Fibonacci sequence.
2. **Iterative Calculation**: For values of n greater than 1, use a loop to compute each Fibonacci number up to the n-th number. We maintain two variables, `a` and `b`, to keep track of the two most recent Fibonacci numbers. In each iteration, compute the next Fibonacci number as the sum of `a` and `b`, then update `a` and `b` to the next pair of numbers.
3. **Efficiency**: This approach runs in O(n) time complexity because each iteration from 2 to n performs a constant amount of work. The space complexity is O(1) since we only
```


### code-32768

**native:**

```text
To solve this problem, we need to write a function that returns the n-th Fibonacci number iteratively. The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, starting from 0 and 1. The function should efficiently compute the n-th Fibonacci number using an iterative approach.

### Approach
The approach to solve this problem involves using an iterative method to compute the Fibonacci sequence. Here's a step-by-step breakdown of the approach:

1. **Base Cases**: If n is 0, return 0. If n is 1, return 1. These are the starting points of the Fibonacci sequence.
2. **Iterative Calculation**: For values of n greater than 1, use a loop to compute each Fibonacci number up to the n-th number. We maintain two variables, `a` and `b`, to keep track of the two most recent Fibonacci numbers. In each iteration, compute the next Fibonacci number as the sum of `a` and `b`, then update `a` and `b` to the next pair of numbers.
3. **Efficiency**: This approach runs in O(n) time complexity because it iterates from 2 to n, performing a constant amount of work in each iteration. The space
```

**cublas:**

```text
To solve this problem, we need to write a function that returns the n-th Fibonacci number iteratively. The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, starting from 0 and 1. The function should efficiently compute the n-th Fibonacci number using an iterative approach.

### Approach
The approach to solve this problem involves using an iterative method to compute the Fibonacci sequence. Here are the key steps:

1. **Base Cases**: If n is 0, return 0. If n is 1, return 1. These are the starting points of the Fibonacci sequence.
2. **Iterative Calculation**: For values of n greater than 1, use a loop to compute each Fibonacci number up to the n-th number. We maintain two variables, `a` and `b`, to keep track of the two most recent Fibonacci numbers. In each iteration, compute the next Fibonacci number as the sum of `a` and `b`, then update `a` and `b` to the next pair of numbers.
3. **Efficiency**: This approach runs in O(n) time complexity because each iteration from 2 to n performs a constant amount of work. The space complexity is O(1) since we only
```


### arith-512

**native:**

```text
**Step‑by‑step solution**

1. **Start time:** The train leaves at **3:40 pm**.  
2. **Trip duration:** The journey lasts **2 hours 35 minutes**.  
3. **Add the hours:**  
   - 3:40 pm + 2 hours = **5:40 pm**.  
4. **Add the minutes:**  
   - 5:40 pm + 35 minutes = **6:15 pm**.  

**Result:** The train arrives at **6:15 pm**.
```

**cublas:**

```text
**Step‑by‑step solution**

1. **Start time:** The train leaves at **3:40 pm**.  
2. **Trip duration:** The journey lasts **2 hours 35 minutes**.  
3. **Add the hours:**  
   - 3:40 pm + 2 hours = **5:40 pm**.  
4. **Add the minutes:**  
   - 5:40 pm + 35 minutes = **6:15 pm**.  

**Result:** The train arrives at **6:15 pm**.
```


### arith-2560

**native:**

```text
**Step‑by‑step solution**

1. **Start time:** The train leaves at **3:40 pm**.  
2. **Trip duration:** The journey lasts **2 hours 35 minutes**.  
3. **Add the hours:**  
   - 3:40 pm + 2 hours = **5:40 pm**.  
4. **Add the minutes:**  
   - 5:40 pm + 35 minutes = **6:15 pm**.  

**Result:** The train arrives at **6:15 pm**.
```

**cublas:**

```text
**Step‑by‑step solution**

1. **Start time:** The train leaves at **3:40 pm**.  
2. **Trip duration:** The journey lasts **2 hours 35 minutes**.  
3. **Add the hours:**  
   - 3:40 pm + 2 hours = **5:40 pm**.  
4. **Add the minutes:**  
   - 5:40 pm + 35 minutes = **6:15 pm**.  

**Result:** The train arrives at **6:15 pm**.
```


### arith-4096

**native:**

```text
**Step‑by‑step solution**

1. **Start time:** The train leaves at **3:40 pm**.  
2. **Trip duration:** The journey lasts **2 hours 35 minutes**.  
3. **Add the hours:**  
   - 3:40 pm + 2 hours = **5:40 pm**.  
4. **Add the minutes:**  
   - 5:40 pm + 35 minutes = **6:15 pm**.  

**Result:** The train arrives at **6:15 pm**.
```

**cublas:**

```text
**Step‑by‑step solution**

1. **Start time:** The train leaves at **3:40 pm**.  
2. **Trip duration:** The journey lasts **2 hours 35 minutes**.  
3. **Add the hours:**  
   - 3:40 pm + 2 hours = **5:40 pm**.  
4. **Add the minutes:**  
   - 5:40 pm + 35 minutes = **6:15 pm**.  

**Result:** The train arrives at **6:15 pm**.
```


### arith-32768

**native:**

```text
**Step‑by‑step solution**

1. **Start time:** The train leaves at **3:40 pm**.  
2. **Trip duration:** The journey lasts **2 hours 35 minutes**.  
3. **Add the hours:**  
   - 3:40 pm + 2 hours = **5:40 pm**.  
4. **Add the minutes:**  
   - 5:40 pm + 35 minutes = **6:15 pm**.  

**Result:** The train arrives at **6:15 pm**.
```

**cublas:**

```text
**Step‑by‑step solution**

1. **Start time:** The train leaves at **3:40 pm**.  
2. **Trip duration:** The journey lasts **2 hours 35 minutes**.  
3. **Add the hours:**  
   - 3:40 pm + 2 hours = **5:40 pm**.  
4. **Add the minutes:**  
   - 5:40 pm + 35 minutes = **6:15 pm**.  

**Result:** The train arrives at **6:15 pm**.
```


### poem-512

**native:**

```text
Ice clings to the stone, a steadfast light,
The wind howls through the frozen night,
A beacon for the wandering soul,
In winter's hush, it holds its promise.
```

**cublas:**

```text
Ice clings to the stone, a steadfast light,
The wind howls through the frozen night,
A beacon for the wandering soul,
In winter's hush, it holds its promise.
```


### poem-2560

**native:**

```text
Ice clings to the stone, a steadfast light,
The wind howls through the frozen night,
A beacon for the wandering soul,
In winter's hush, it holds its promise.
```

**cublas:**

```text
Ice clings to the stone, a steadfast light,
The wind howls through the frozen night,
A beacon for the wandering soul,
In winter's hush, it holds its promise.
```


### poem-4096

**native:**

```text
Ice clings to the stone, a steadfast light,
The wind howls through the frozen night,
A beacon for the wandering soul,
In winter's hush, it holds its promise.
```

**cublas:**

```text
Ice clings to the stone, a steadfast light,
The wind howls through the frozen night,
A beacon for the wandering soul,
In winter's hush, it holds its promise.
```


### poem-32768

**native:**

```text
Ice clings to the stone, a steadfast light,
The wind howls through the frozen night,
A beacon for the wandering soul,
In winter's hush, it holds its promise.
```

**cublas:**

```text
Ice clings to the stone, a steadfast light,
The wind howls through the frozen night,
A beacon for the wandering soul,
In winter's hush, it holds its promise.
```


### memory-512

**native:**

```text
Your name is Priya. Before winter, you should check the hive for signs of disease, such as mold or pests, and ensure the bees have enough food and water.
```

**cublas:**

```text
Your name is Priya. Before winter, you should check the hive for signs of disease, inspect the bees for signs of illness, and make sure the hive is well-ventilated.
```


### memory-2560

**native:**

```text
Your name is Priya. Before winter, you should check the hive for signs of disease, such as mold or pests, and ensure the bees have enough food and water.
```

**cublas:**

```text
Your name is Priya. Before winter, you should check the hive for signs of disease, inspect the bees for signs of illness, and make sure the hive is well-ventilated.
```


### memory-4096

**native:**

```text
Your name is Priya. Before winter, you should check the hive for signs of disease, such as mold or pests, and ensure the bees have enough food and water.
```

**cublas:**

```text
Your name is Priya. Before winter, you should check the hive for signs of disease, inspect the bees for signs of illness, and make sure the hive is well-ventilated.
```


### memory-32768

**native:**

```text
Your name is Priya. Before winter, you should check the hive for signs of disease, such as mold or pests, and ensure the bees have enough food and water.
```

**cublas:**

```text
Your name is Priya. Before winter, you should check the hive for signs of disease, inspect the bees for signs of illness, and make sure the hive is well-ventilated.
```


### doc-2560

**native:**

```text
**Passphrase:** violet harbor 42  

The text is a biographical overview of the Chinese poet Du Fu (712–770), covering his early life, career, and literary significance, and then lists his film and television/ theatre work.
```

**cublas:**

```text
**Passphrase:** violet harbor 42  

The text is a biographical overview of the Chinese poet Du Fu (712–770), covering his early life, career, and literary significance, and then lists his film and television/ theatre work.
```


### doc-4096

**native:**

```text
**Archivist's secret passphrase:** violet harbor 42  

**Summary:** The text provides a detailed biography of the Tang dynasty Chinese poet Du Fu, covering his early life, career as a civil servant, experiences during the An Lushan Rebellion, his time in Chengdu, and his later years. It also includes a brief overview of Du Fu, highlighting his status as one of China's greatest poets and his influence on Chinese and Japanese literary culture.
```

**cublas:**

```text
**Archivist's secret passphrase:** violet harbor 42  

**Summary:** The text provides a detailed biography of the Chinese poet Du Fu, covering his early life, the An Lushan Rebellion, his move to Chengdu, and his later years, while also offering a brief overview of Du Fu as one of the greatest Chinese poets, known for his influential works and themes drawn from his life experiences.
```


### doc-32768

**native:**

```text
**Archivist's secret passphrase:**  
Violet harbor 42  

**Summary of the text:**  
The text provides a comprehensive biographical and career overview of Robert Boulter, an English actor known for his roles in television series like *The Bill* and *Judge John Deed*, as well as theatre productions such as *Herons* and *Mercury Fur*. It details his film appearances, including *Daylight Robbery* and *Donkey Punch*, and highlights his critical acclaim for performances in works like *Citizenship* and *How to Curse*. The text also includes a separate section on the 1933 Treasure Coast hurricane, describing its formation, path, impacts on Florida and the Bahamas, and aftermath.
```

**cublas:**

```text
**Archivist's secret passphrase:**  
**"violet harbor 42"**

**Summary of the text:**  
The text provides a comprehensive biographical and career overview of Robert Boulter, an English actor, detailing his roles in television, theatre, and film from 2000 to 2011, including notable performances in *The Bill*, *Herons*, *The Long Firm*, and *Citizenship*. It also includes a detailed biography of Du Fu, a Tang dynasty Chinese poet, covering his early life, military service, poetic career, and influence on Chinese and Japanese literature. Additionally, the text discusses the 1933 Treasure Coast hurricane, its meteorological history, and the Second Battle of Naktong Bulge during the Korean War.
```


Every case produced output, and every text equals the HF tokenizer's decode of its ids.
