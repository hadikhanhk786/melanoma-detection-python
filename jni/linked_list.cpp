/****************************************************************************
**
** Copyright (C) 2010-2011 Fabien Bessy.
** All rights reserved.
** Contact: fabien.bessy@gmail.com
**
** This file is part of Ofeli.
**
** You may use this file under the terms of the CeCILL license version 2 as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of Fabien Bessy nor the names of its contributors
**     may be used to endorse or promote products derived from this
**     software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
**                    For all legal intents and purposes.
**
****************************************************************************/

#include <iostream> // for the function display()
#include "linked_list.hpp"

namespace ofeli
{

const int List::sentinel_elem = -2147483648; // min value with a 32 bits architecture

List::List()
{
    head = NULL;
    push_front(sentinel_elem); // creates the sentinal node after the last node to allow to erase the last node

    begin(); // current = head;
}

List::List(const List& list)
{
    head = NULL;
    push_front(sentinel_elem); // creates the sentinal node after the last node to allow to erase the last node

    begin();
    for( const_Link iterator = list.get_begin(); !iterator->end(); iterator = iterator->get_next() )
    {
        insert_after( iterator->get_elem() );
    }

    begin();
}

List& List::operator=(const List& list)
{
    if( &list != this ) // no auto-affectation
    {
        head = NULL;
        push_front(sentinel_elem); // creates the sentinal node after the last node to allow to erase the last node

        begin();
        for( const_Link iterator = list.get_begin(); !iterator->end(); iterator = iterator->get_next() )
        {
            insert_after( iterator->get_elem() );
        }

        begin();
    }

    return *this;
}

List::~List()
{
    Link new_head;

    while( head != NULL ) // while the list is not empty (including the sentinel node)
    {
        new_head = head->get_next();
        delete head; // delete the first node
        head = new_head;
    }
}

void List::clear()
{
    Link new_head;

    while( head->get_next() != NULL ) // while the list is not empty (without including the sentinel node)
    {
        new_head = head->get_next();
        delete head; // delete the first node
        head = new_head;
    }
    begin();

    return;
}

void List::begin()
{
    current = head;

    return;
}

void List::next()
{
    current = current->get_next();

    return;
}

bool List::end() const
{
    if( current->get_next() == NULL ) // if the node is the sentinel
    {
        return true;
    }
    else
    {
        return false;
    }
}

void List::push_front(int elem) // inserts a new element at the beginning of the list
{
    Link new_head = new Node(elem, head); // to link the new head node to the former head node

    head = new_head; // update of the head link

    return;
}

void List::insert_after(int elem)
{
    if( !empty() )
    {
        Link new_node = new Node( elem, current->get_next() );
        current->set_next(new_node);
        current = new_node;
    }
    else
    {
        push_front(elem);
        begin();
    }

    return;
}

void List::erase() // erase the node at the current position
{
    Link next_node = current->get_next();

    // copy the next node to the current node
    *current = *next_node; //  <==> current->set_elem( next_node->get_elem() ); current->set_next( next_node->get_next() );

    delete next_node; // erase the next node

    return;
}

void List::switch_current(List& list)
{
    Link node = list.get_current_node();

    int temp = node->get_elem();

    Link next_node = node->get_next();

    // copy the next node to the node
    *node = *next_node; // <==> node->set_elem( next_node->get_elem() ); node->set_next( next_node->get_next() );

    next_node->set_elem_next(temp,head);

    head = next_node; // update of the head link

    return;
}

int List::get_current() const
{
    return current->get_elem();
}

List::Link List::get_current_node() const
{
    return current;
}

List::Link List::get_begin() const
{
    return head;
}

bool List::empty() const
{
    if( head->end() ) // if the head node is the sentinel
    {
        return true;
    }
    else
    {
        return false;
    }
}

// get the size in O(n)
int List::size() const
{
    int size = 0;

    for( const_Link iterator = get_begin(); !iterator->end(); iterator = iterator->get_next() ) // scan through the list
    {
        size++;
    }

    return size;
}

bool List::operator==(const List& list) const
{
    const_Link iterator2 = get_begin();

    for( const_Link iterator1 = list.get_begin(); !iterator1->end(); iterator1 = iterator1->get_next() ) // scan through the list
    {
        if( !iterator2->end() )
        {
            if( iterator1->get_elem() != iterator2->get_elem() )
            {
                return false;
            }

            iterator2 = iterator2->get_next();
        }
        else
        {
            return false;
        }
    }

    if( iterator2->end() )
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool List::operator!=(const List& list) const
{
    return !( *this == list );
}

void List::display() const
{
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "      Linked list   " << std::endl;
    std::cout << " --------------------- " << std::endl;
    std::cout << "  Position | Value " << std::endl;
    std::cout << " --------------------- " << std::endl;

    int position = 1;
    for( const_Link iterator = get_begin(); !iterator->end(); iterator = iterator->get_next() ) // scan through the list
    {
        if( position < 10 )
        {
            std::cout << "     " << position << "     |   " << iterator->get_elem() << std::endl;
        }

        if( position >= 10 && position < 100 )
        {
            std::cout << "     " << position << "    |   " << iterator->get_elem() << std::endl;
        }

        if( position >= 100 && position < 1000 )
        {
            std::cout << "     " << position << "   |   " << iterator->get_elem() << std::endl;
        }

        if( position >= 1000 )
        {
            std::cout << "     " << position << "  |   " << iterator->get_elem() << std::endl;
        }

        position++;
    }

    std::cout << " --------------------- " << std::endl;

    return;
}





List::Node::Node(int elem1, List::Link next1) : elem(elem1), next(next1)
{
}

int List::Node::get_elem() const
{
    return elem;
}

List::Link List::Node::get_next() const
{
    return next;
}

void List::Node::set_elem(int elem1)
{
    elem = elem1;
    return;
}

void List::Node::set_next(List::Link next1)
{
    next = next1;
    return;
}

void List::Node::set_elem_next(int elem1, List::Link next1)
{
    elem = elem1;
    next = next1;
    return;
}

bool List::Node::end() const
{
    if( next == NULL ) // if the node is the sentinel
    {
        return true;
    }
    else
    {
        return false;
    }
}

}
