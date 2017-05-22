#ifndef LINKED_LIST_HPP
#define LINKED_LIST_HPP

namespace ofeli
{

class List
{

public :

    //! Constructor.
    List();

    //! Destructor. All the elements in the list container are dropped (including the sentinel node) : their destructors are called, and then they are removed from the list container, leaving it with a size of 0.
    ~List();

    //! Copy constructor.
    List(const List& list);

    //! Assignment operator overloading.
    List& operator=(const List& list);

    //!  \a Egal \a to operator overloading.
    bool operator==(const List& list) const;

    //! \a Not \a egal \a to operator overloading.
    bool operator!=(const List& list) const;

    //! Node class : an element with a pointer on the next node.
    class Node;

    //! Link is a pointer to a Node.
    typedef Node* Link;

    //! const_Link is a pointer to a Node.
    typedef const Node* const_Link;

    //! Sets the current link to the head link.
    void begin();
    //! Sets the current link to the next link.
    void next();
    //! Checks if he current link points to the last node.
    bool end() const;

    //! Inserts a new element at the beginning of the list.
    void push_front(int elem);

    //! Inserts a new element after the current position of the list.
    void insert_after(int elem);

    //! Removes a current node of a list and inserts at the beginning of this list. Equivalent to erase() and push_front() without use of the operators \a delete and \a new.
    void switch_current(List& list);

    //! Removes from the list container the element at the current position.
    void erase();
    //! All the elements in the list container are dropped (except the sentinel node) : their destructors are called, and then they are removed from the list container, leaving it with a size of 0.
    void clear();

    //! Gets the element at the current position.
    int get_current() const;

    //! Gets the link at the current position.
    Link get_current_node() const;

    //! This class implements a node of the class List. It is composed by an integer element and a pointer on the next node.
    class Node
    {

    public :

        //! Constructor.
        Node(int elem1, Link next1);
        //! Getter of the element.
        int get_elem() const;
        //! Getter of the link to the next node.
        Link get_next() const;

        //! Setter of the element.
        void set_elem(int elem1);
        //! Setter of the link to the next node.
        void set_next(Link next1);
        //! Setter of the element and the link.
        void set_elem_next(int elem1, Link next1);

        //! Checks if the node is the sentinel node.
        bool end() const;

    private :

        //! Element storage.
        int elem;
        //! Pointer to the next node.
        Link next;

    };

    //! Gets the head of the list.
    Link get_begin() const;

    //! Returns whether the list container is empty, i.e. whether its size is 0.
    bool empty() const;

    //! Returns the number of elements in the list without counting the sentinel node.
    //! The complexity of this function is in \a O(n) so if you want to get often the size of the list, you should update the size in a variable after each modification.
    int size() const;

    //! Displays the linked list in the standard output.
    void display() const;

private :

    //! Head of the list.
    Link head;

    //! Current position of the list.
    Link current;

    //! Value of the last sentinel node to allow to remove the last true node in \a 0(1).
    static const int sentinel_elem;

};

}

#endif

//! \class ofeli::List
//! This class implements a singly linked list used by the implementation of Shi and Karl' algorithm. The elements stored are integers that correspond of the offset of the level set function buffer.

/**
 * \example linked_list.cpp
 * \code
 * // How to use the linked list ?
 *
 * ofeli::List list; // list is empty
 * list.display();   // display nothing in the standard output
 * \endcode
 * \code
 * Standard output result :
 *
 *      Linked list
 * ---------------------
 *  Position | Value
 * ---------------------
 * ---------------------
 * \endcode
 * \code
 * list.push_front(99); // list = 99
 * list.push_front(99); // list = 99|99
 * list.push_front(2);  // list = 2|99|99
 * list.push_front(5);  // list = 5|2|99|99
 * list.push_front(99); // list = 99|5|2|99|99
 * list.display();      // display the list in the standard output;
 * \endcode
 * \code
 * Standard output result :
 *
 *      Linked list
 * ---------------------
 *  Position | Value
 * ---------------------
 *     1     |   99
 *     2     |   5
 *     3     |   2
 *     4     |   99
 *     5     |   99
 * ---------------------
 * \endcode
 * \code
 * list.push_front(1);  // list = 1|99|5|2|99|99
 * list.push_front(99); // list = 99|1|99|5|2|99|99
 * list.display();      // display the list in the standard output;
 * \endcode
 * \code
 * Standard output result :
 *
 *      Linked list
 * ---------------------
 *  Position | Value
 * ---------------------
 *     1     |   99
 *     2     |   1
 *     3     |   99
 *     4     |   5
 *     5     |   2
 *     6     |   99
 *     7     |   99
 * ---------------------
 * \endcode
 * \code
 * // Removes all values of the list strictly greater than 50.
 *
 * // void begin() : the current position is the head of the list
 * // bool end() : true if the current position is the tail of the list
 *
 * for( list.begin(); !list.end(); )
 * {
 *     if( list.get_current() > 50 ) // if the value at the current position is greater than 50
 *     {
 *         list.erase(); // erase the value at the current position, the current position is the next element, now
 *     }
 *     else
 *     {
 *         list.next(); // similar to "++iterator;" for the STL list, moves to the next element in the list.
 *     }
 * }
 * list.display(); // display the list in the standard output;
 * \endcode
 * \code
 * Standard output result :
 *
 *      Linked list
 * ---------------------
 *  Position | Value
 * ---------------------
 *     1     |   1
 *     2     |   5
 *     3     |   2
 * ---------------------
 * \endcode
 * \code
 * list.clear(); // list is empty
 * list.push_front(63);
 * list.display(); // display the list in the standard output;
 * \endcode
 * \code
 * Standard output result :
 *
 *      Linked list
 * ---------------------
 *  Position | Value
 * ---------------------
 *     1     |   63
 * ---------------------
 *
 * \endcode
 * \code
 * // If you want to read a list without modifying it. You should use const_Link.
 * for( const_Link iterator = list.get_begin(); !iterator->end(); iterator = iterator->get_next() ) // scan through the list
   {
       std::cout << "value = " << iterator->get_elem() << std::endl;
   }
 * \endcode
 */

