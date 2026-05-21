"""
Enrich central.db with keywords, descriptions, and subtopics for all topics.
Run: .env/bin/python populate_db.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "central.db")

# Full topic data: topic_name -> {description, keywords, key_concepts, example_questions}
TOPIC_DATA = {
    # ===== Mathematics > Algebra =====
    "Linear Equations": {
        "description": "Equations where variables have degree 1. Solving for unknowns using addition, subtraction, multiplication, and division properties of equality.",
        "keywords": "linear equation,solve,variable,unknown,system of equations,substitution,elimination,graph,slope intercept,standard form,one variable,two variables,simultaneous equations",
        "key_concepts": "Variables and constants,Balancing equations,Substitution method,Elimination method,Graphical solution,Slope-intercept form y=mx+b,Standard form ax+by=c,Infinitely many solutions,No solution,Unique solution",
        "example_questions": "How to solve 2x+3=7?,Solve the system x+y=5 and x-y=1,What is the slope-intercept form?,How to graph a linear equation?",
    },
    "Quadratic Equations": {
        "description": "Polynomial equations of degree 2 in the form ax²+bx+c=0. Solved by factoring, completing the square, or the quadratic formula.",
        "keywords": "quadratic,equation,formula,discriminant,roots,solutions,factoring,parabola,completing the square,x intercepts,vertex,axis of symmetry,coefficient",
        "key_concepts": "Standard form ax²+bx+c=0,Quadratic formula x=(-b±√(b²-4ac))/2a,Discriminant b²-4ac,Factoring,Completing the square,Sum of roots,Product of roots,Parabola graph,Vertex form,Nature of roots",
        "example_questions": "What is the quadratic formula?,How does the discriminant affect roots?,How to factor x²-5x+6=0?,What is completing the square?",
    },
    "Polynomials": {
        "description": "Expressions with variables raised to whole number powers. Operations include addition, subtraction, multiplication, division, and factoring.",
        "keywords": "polynomial,term,coefficient,degree,monomial,binomial,trinomial,factor,expand,roots,zeros,division,remainder theorem,factor theorem",
        "key_concepts": "Terms and coefficients,Degree of polynomial,Monomial binomial trinomial,Addition and subtraction,Multiplication (FOIL),Factoring techniques,Remainder theorem,Factor theorem,Rational root theorem,Polynomial long division",
        "example_questions": "What are polynomials?,How to factor a polynomial?,What is the remainder theorem?,How to find zeros of a polynomial?",
    },
    # ===== Mathematics > Calculus =====
    "Limits": {
        "description": "The value a function approaches as the input approaches a given point. Foundation of calculus for defining derivatives and integrals.",
        "keywords": "limit,approach,tends to,continuity,infinity,one-sided limit,two-sided limit,indeterminate form,L'Hopital,squeeze theorem,infinite limit,limit at infinity",
        "key_concepts": "Intuitive idea of a limit,One-sided limits (left and right),Limit laws and properties,Continuity and limits,Indeterminate forms 0/0 and ∞/∞,L'Hopital's rule,Squeeze theorem,Limits at infinity,Limit of trigonometric functions,Piecewise function limits",
        "example_questions": "What is a limit in calculus?,How to evaluate lim(x→2) of (x²-4)/(x-2)?,What is L'Hopital's rule?,When is a function continuous?",
    },
    "Derivatives": {
        "description": "Rate of change of a function with respect to a variable. Measures the slope of the tangent line at any point on a curve.",
        "keywords": "derivative,differentiation,rate of change,slope,tangent,chain rule,product rule,quotient rule,power rule,second derivative,implicit differentiation,related rates",
        "key_concepts": "Definition as limit of difference quotient,Power rule,Product rule,Quotient rule,Chain rule,Derivatives of trig functions,Derivatives of exponential and log,Second derivative and concavity,Implicit differentiation,Related rates problems",
        "example_questions": "How to find the derivative of x²?,What is the chain rule?,How to differentiate sin(x)?,What does the second derivative tell us?",
    },
    "Integrals": {
        "description": "The reverse of differentiation. Computes areas under curves, accumulated quantities, and antiderivatives.",
        "keywords": "integral,integration,antiderivative,area under curve,definite integral,indefinite integral,fundamental theorem,Riemann sum,substitution,by parts,area,volume",
        "key_concepts": "Indefinite integrals (antiderivatives),Definite integrals and area,Fundamental theorem of calculus,Integration by substitution,Integration by parts,Standard integrals,Riemann sums,Numerical integration,Applications: area and volume,Improper integrals",
        "example_questions": "What is integration?,How to compute ∫x² dx?,What is the fundamental theorem of calculus?,How to use substitution?",
    },
    # ===== Mathematics > Geometry =====
    "Triangles": {
        "description": "Three-sided polygons. Classification by sides (equilateral, isosceles, scalene) and angles (acute, right, obtuse). Congruence and similarity rules.",
        "keywords": "triangle,equilateral,isosceles,scalene,acute,right,obtuse,congruence,similarity,SSS,SAS,ASA,AAS,HL,Pythagorean theorem,area,perimeter,height,median",
        "key_concepts": "Classification by sides,Classification by angles,Angle sum property (180°),Congruence criteria SSS SAS ASA AAS,Similarity criteria AA SAS SSS,Pythagorean theorem,Area formulas,Median and centroid,Altitude and orthocenter,Bisectors",
        "example_questions": "What are the types of triangles?,How to prove triangle congruence?,What is the Pythagorean theorem?,How to find area of a triangle?",
    },
    "Circles": {
        "description": "Set of all points equidistant from a center. Properties include radius, diameter, circumference, arc length, and sector area.",
        "keywords": "circle,radius,diameter,circumference,arc,sector,chord,tangent,secant,pi,central angle,inscribed angle,equation of circle,arc length,sector area",
        "key_concepts": "Standard equation (x-h)²+(y-k)²=r²,Radius and diameter,Circumference = 2πr,Area = πr²,Arc length and sector area,Central and inscribed angles,Chord properties,Tangent line properties,Cyclic quadrilaterals,Inscribed angle theorem",
        "example_questions": "What is the equation of a circle?,How to find arc length?,What is a tangent to a circle?,How to find sector area?",
    },
    "Coordinate Geometry": {
        "description": "Study of geometry using coordinate systems. Combines algebra and geometry to solve problems about points, lines, and shapes on a plane.",
        "keywords": "coordinate,plane,Cartesian,distance,midpoint,slope,section formula,parallel,perpendicular,line equation,intersection,origin,x-axis,y-axis",
        "key_concepts": "Cartesian coordinate system,Distance formula,Midpoint formula,Slope of a line,Section formula,Parallel and perpendicular lines,Equation of a line,Intersection of lines,Area using coordinates,Distance from point to line",
        "example_questions": "How to find distance between two points?,What is the midpoint formula?,How to find slope from two points?,What is the section formula?",
    },
    # ===== Physics > Mechanics =====
    "Newton's Laws": {
        "description": "Three fundamental laws of motion that describe relationship between forces and motion. Foundation of classical mechanics.",
        "keywords": "Newton,law,motion,inertia,force,acceleration,mass,action,reaction,F=ma,friction,gravity,weight,normal force,tension,net force,free body diagram",
        "key_concepts": "First law: inertia,Second law: F=ma,Third law: action-reaction,Force and acceleration,Mass vs weight,Free body diagrams,Normal force,Friction (static and kinetic),Tension in strings,Newton's law of gravitation",
        "example_questions": "Explain Newton's laws of motion,What is inertia?,What is F=ma?,Give an example of action-reaction pair",
    },
    "Work & Energy": {
        "description": "Work is force applied over distance. Energy is the capacity to do work. Conservation of energy states energy cannot be created or destroyed.",
        "keywords": "work,energy,kinetic,potential,conservation,power,joule,watt,force,displacement,gravitational PE,spring PE,mechanical energy,work-energy theorem",
        "key_concepts": "Work = Force × displacement × cosθ,Kinetic energy = ½mv²,Gravitational PE = mgh,Conservation of mechanical energy,Work-energy theorem,Power = Work/time,Conservative and non-conservative forces,Spring potential energy,Elastic and inelastic collisions,Energy transformations",
        "example_questions": "What is work in physics?,Explain conservation of energy,What is kinetic energy?,How to calculate power?",
    },
    "Momentum": {
        "description": "Product of mass and velocity (p=mv). Conservation of momentum is a fundamental law. Used to analyze collisions and explosions.",
        "keywords": "momentum,impulse,collision,elastic,inelastic,conservation,velocity,mass,impulse-momentum theorem,recoil,explosion,perfectly inelastic,coefficient of restitution",
        "key_concepts": "Linear momentum p=mv,Impulse = Force × time,Impulse-momentum theorem,Conservation of momentum,Elastic collisions,Inelastic collisions,Perfectly inelastic collisions,Coefficient of restitution,Recoil problems,Two-dimensional collisions",
        "example_questions": "What is momentum?,Explain elastic vs inelastic collision,What is impulse?,How is momentum conserved in a collision?",
    },
    # ===== Physics > Thermodynamics =====
    "Heat Transfer": {
        "description": "Transfer of thermal energy between systems. Three modes: conduction (direct contact), convection (fluid movement), radiation (electromagnetic waves).",
        "keywords": "heat,transfer,conduction,convection,radiation,thermal,temperature,conductivity,insulator,conductor,thermal equilibrium,heat flow,Newton's law of cooling",
        "key_concepts": "Conduction through solids,Convection in fluids,Radiation via electromagnetic waves,Thermal conductivity,Conductors and insulators,Newton's law of cooling,Stefan-Boltzmann law,Heat transfer rate,Fourier's law,Thermal equilibrium",
        "example_questions": "What are the modes of heat transfer?,How does conduction work?,What is convection?,How does the sun transfer heat to Earth?",
    },
    "Laws of Thermodynamics": {
        "description": "Fundamental laws governing heat, work, and energy. Zeroth: thermal equilibrium. First: energy conservation. Second: entropy increases. Third: absolute zero.",
        "keywords": "thermodynamics,law,zeroth,first,second,third,internal energy,heat,work,entropy,absolute zero,enthalpy,Carnot engine,efficiency,isolated system",
        "key_concepts": "Zeroth law and thermal equilibrium,First law: ΔU = Q - W,Internal energy,Second law and entropy,Heat engines and efficiency,Carnot cycle,Reversible and irreversible processes,Third law: entropy at absolute zero,Enthalpy,Gibbs free energy",
        "example_questions": "Explain the first law of thermodynamics,What is the second law?,What is a Carnot engine?,What happens at absolute zero?",
    },
    "Entropy": {
        "description": "Measure of disorder or randomness in a system. Always increases in an isolated system (second law of thermodynamics).",
        "keywords": "entropy,disorder,randomness,second law,thermodynamics,isolated system,microstates,macrostates,Boltzmann,irreversible,reversible,heat death,disorder",
        "key_concepts": "Entropy as disorder measure,Boltzmann formula S=kB ln(W),Entropy and second law,Entropy in reversible vs irreversible processes,Entropy of the universe always increases,Statistical interpretation,Microstates and macrostates,Heat death of universe,Free energy and entropy,Entropy changes in phase transitions",
        "example_questions": "What is entropy?,How does entropy relate to disorder?,Can entropy decrease locally?,What is the Boltzmann formula?",
    },
    # ===== Physics > Optics =====
    "Reflection": {
        "description": "Bouncing of light off a surface. Law of reflection: angle of incidence equals angle of reflection. Types: specular (smooth) and diffuse (rough).",
        "keywords": "reflection,mirror,angle of incidence,angle of reflection,normal,specular,diffuse,plane mirror,concave,convex,image,virtual image,real image,focal point",
        "key_concepts": "Law of reflection,Angle of incidence = angle of reflection,Normal to the surface,Specular vs diffuse reflection,Plane mirror images,Concave mirror,Convex mirror,Image formation,Ray diagrams,Focal point and center of curvature",
        "example_questions": "What is the law of reflection?,How do mirrors work?,What is a virtual image?,How does a concave mirror focus light?",
    },
    "Refraction": {
        "description": "Bending of light when passing from one medium to another due to change in speed. Governed by Snell's law: n₁sinθ₁ = n₂sinθ₂.",
        "keywords": "refraction,Snell's law,refractive index,bending,light,speed,medium,denser,rarer,total internal reflection,critical angle,optical density,prism,dispersion",
        "key_concepts": "Snell's law n₁sinθ₁ = n₂sinθ₂,Refractive index,Refraction from denser to rarer medium,Refraction from rarer to denser medium,Total internal reflection,Critical angle,Dispersion through prism,Apparent depth,Real and apparent position,Lens maker equation",
        "example_questions": "What is refraction?,Explain Snell's law,What is total internal reflection?,Why does a prism split white light?",
    },
    "Lenses": {
        "description": "Transparent optical devices that refract light to converge or diverge. Convex lenses converge (positive focal length), concave lenses diverge (negative).",
        "keywords": "lens,convex,concave,converging,diverging,focal length,focus,image,object,magnification,lens formula,1/f=1/v+1/u,power,optical center,principal axis",
        "key_concepts": "Convex lens (converging),Concave lens (diverging),Principal axis and optical center,Focal point and focal length,Lens formula 1/f = 1/v - 1/u,Magnification m = v/u,Image formation ray diagrams,Real vs virtual images,Power of lens P=1/f,Combination of lenses",
        "example_questions": "How do convex lenses work?,What is focal length?,What is the lens formula?,How to find image position?",
    },
    # ===== CS > Programming Fundamentals =====
    "Variables & Data Types": {
        "description": "Variables store data in named containers. Data types define what kind of data a variable holds: integers, floats, strings, booleans.",
        "keywords": "variable,data type,integer,float,string,boolean,declare,assign,value,type,constant,literal,type casting,primitive,reference,mutability",
        "key_concepts": "Variable declaration and assignment,Integers and floating point,Strings and string operations,Boolean logic,Type casting and conversion,Constants and literals,Dynamic vs static typing,Mutability,Variable scope,Type checking",
        "example_questions": "What are variables?,Explain data types in Python,What is the difference between int and float?,How does type casting work?",
    },
    "Control Flow": {
        "description": "Determines the order in which statements execute. Includes conditional statements (if-else) and loops (for, while).",
        "keywords": "control flow,if,else,elif,for loop,while loop,loop,condition,iteration,break,continue,pass,nested loop,range,Boolean expression,branching",
        "key_concepts": "If-elif-else statements,Comparison operators,Logical operators and or not,For loops with range,While loops,Break and continue,Nested loops,Loop control,Infinite loops and avoidance,Match-case (Python 3.10+)",
        "example_questions": "What is a for loop?,How do if-else statements work?,What is the difference between for and while?,How to use break and continue?",
    },
    "Functions": {
        "description": "Reusable blocks of code that perform a specific task. Take inputs (parameters), process them, and return outputs.",
        "keywords": "function,def,parameter,argument,return,call,invoke,scope,local,global,recursive,lambda,docstring,pass by value,pass by reference,inner function",
        "key_concepts": "Function definition with def,Parameters and arguments,Return values,Default parameters,Keyword arguments,Variable scope (local vs global),Recursive functions,Lambda functions,Docstrings,Higher-order functions",
        "example_questions": "What is a function?,How to write a recursive function?,What is a lambda function?,What is variable scope?",
    },
    # ===== CS > Data Structures =====
    "Arrays": {
        "description": "Contiguous memory locations storing elements of the same type. Support O(1) random access by index.",
        "keywords": "array,index,element,contiguous,memory,access,insert,delete,search,reverse,sort,traversal,linear,static,dynamic,2D array,matrix",
        "key_concepts": "Contiguous memory allocation,Index-based O(1) access,Static vs dynamic arrays,Insertion and deletion,Searching in arrays,Reversing an array,Sorting arrays,2D arrays and matrices,Array vs list,Time complexity of operations",
        "example_questions": "What is an array?,How to reverse an array?,What is the time complexity of array access?,How to search in an array?",
    },
    "Linked Lists": {
        "description": "Linear data structure where elements (nodes) point to the next node. Dynamic size, efficient insertion/deletion at any position.",
        "keywords": "linked list,node,next pointer,head,singly,doubly,circular,insert,delete,reverse,detect cycle,fast slow pointer,Floyd,memory,sequential access",
        "key_concepts": "Node structure (data + next pointer),Singly linked list,Doubly linked list,Circular linked list,Head and tail,Insertion at beginning/end/middle,Deletion operations,Reversing a linked list,Cycle detection (Floyd's algorithm),Linked list vs array",
        "example_questions": "What is a linked list?,How to detect a cycle in a linked list?,How to reverse a linked list?,What is Floyd's cycle detection?",
    },
    "Trees & Graphs": {
        "description": "Trees are hierarchical structures with root and children. Graphs are collections of nodes (vertices) connected by edges.",
        "keywords": "tree,graph,root,node,edge,vertex,binary tree,BST,BFS,DFS,preorder,inorder,postorder,adjacency,weighted,directed,undirected,tree traversal,heap",
        "key_concepts": "Tree terminology (root, leaf, height),Binary tree,Binary search tree (BST),Tree traversals (preorder, inorder, postorder),BFS (breadth-first search),DFS (depth-first search),Graph types (directed, undirected, weighted),Adjacency matrix vs list,Shortest path basics,Heap and priority queue",
        "example_questions": "What is a binary tree?,Explain BFS vs DFS traversal,What is a BST?,How to traverse a tree inorder?",
    },
    # ===== CS > Algorithms =====
    "Sorting": {
        "description": "Arranging elements in a specific order. Various algorithms with different time/space tradeoffs: bubble, merge, quick, heap sort.",
        "keywords": "sort,sorting,bubble,merge,quick,heap,insertion,selection,comparison,time complexity,O(n log n),O(n²),stable,in-place,divide and conquer,partition",
        "key_concepts": "Bubble sort O(n²),Selection sort O(n²),Insertion sort O(n²),Merge sort O(n log n),Quick sort O(n log n) avg,Heap sort O(n log n),Stable vs unstable sorting,In-place sorting,Divide and conquer strategy,Space complexity tradeoffs",
        "example_questions": "How does quicksort work?,What is merge sort?,Which sorting algorithm is fastest?,What does stable sort mean?",
    },
    "Searching": {
        "description": "Finding a target value in a collection. Linear search checks each element O(n). Binary search halves the search space O(log n) on sorted data.",
        "keywords": "search,linear search,binary search,sorted,target,element,O(n),O(log n),divide,halving,comparison,iterative,recursive,hash table,lookup",
        "key_concepts": "Linear search O(n),Binary search O(log n),Binary search prerequisites (sorted data),Binary search algorithm (iterative),Binary search algorithm (recursive),Comparison: linear vs binary,Hash-based O(1) lookup,Search in 2D arrays,Lower and upper bound,Ternary search",
        "example_questions": "What is binary search?,How does linear search work?,When to use binary search over linear?,What is the time complexity of binary search?",
    },
    "Dynamic Programming": {
        "description": "Optimization technique that solves complex problems by breaking them into overlapping subproblems. Stores results to avoid redundant computation.",
        "keywords": "dynamic programming,memoization,tabulation,subproblem,optimal substructure,overlapping subproblems,bottom-up,top-down,cache,table,fibonacci,knapsack,LCS,DP state,recurrence relation",
        "key_concepts": "Optimal substructure property,Overlapping subproblems,Memoization (top-down),Tabulation (bottom-up),Memoization vs tabulation,DP state and transitions,Recurrence relations,Classic: Fibonacci,Classic: Knapsack problem,Classic: Longest common subsequence",
        "example_questions": "What is dynamic programming?,Explain memoization vs tabulation,What is optimal substructure?,How to solve Fibonacci with DP?",
    },
}


def populate():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Add new columns to topics table
    cols = [r[1] for r in cur.execute("PRAGMA table_info(topics)").fetchall()]
    new_cols = {
        "description": "TEXT",
        "keywords": "TEXT",
        "key_concepts": "TEXT",
        "example_questions": "TEXT",
    }
    for col_name, col_type in new_cols.items():
        if col_name not in cols:
            cur.execute(f"ALTER TABLE topics ADD COLUMN {col_name} {col_type}")
            print(f"  Added column: {col_name}")

    # Populate each topic
    updated = 0
    for topic_name, data in TOPIC_DATA.items():
        cur.execute(
            """UPDATE topics SET
                   description = ?, keywords = ?,
                   key_concepts = ?, example_questions = ?
               WHERE topic_name = ?""",
            (
                data["description"],
                data["keywords"],
                data["key_concepts"],
                data["example_questions"],
                topic_name,
            ),
        )
        if cur.rowcount > 0:
            updated += 1
            print(f"  Updated: {topic_name}")
        else:
            print(f"  NOT FOUND: {topic_name}")

    conn.commit()

    # Verify
    total = cur.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    filled = cur.execute(
        "SELECT COUNT(*) FROM topics WHERE keywords IS NOT NULL AND keywords != ''"
    ).fetchone()[0]
    print(f"\n  Topics: {total} total, {filled} enriched")

    # Show sample
    row = cur.execute(
        "SELECT topic_name, keywords, key_concepts FROM topics WHERE topic_id = 1"
    ).fetchone()
    if row:
        print(f"\n  Sample (topic_id=1):")
        print(f"    Name: {row[0]}")
        print(f"    Keywords: {row[1][:80]}...")
        print(f"    Concepts: {row[2][:80]}...")

    conn.close()


if __name__ == "__main__":
    print("Populating central.db with topic data...")
    populate()
    print("\nDone.")
