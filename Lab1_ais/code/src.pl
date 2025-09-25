% Существующие родственные связи.
% -------------------------------------------------------------------
parent(nikolay, andrew).
parent(tamara, andrew).
parent(nikolay, larisa).
parent(tamara, larisa).
parent(larisa, denis).
parent(larisa, angelina).

parent(georgiy, galina).
parent(agafya, galina).
parent(georgiy, tatiana).
parent(agafya, tatiana).
parent(georgiy, vladimir).
parent(agafya, vladimir).

parent(boris, mama).
parent(galina, mama).

parent(andrew, dmitriy).
parent(mama, dmitriy).

parent(vyacheslav, ludmila).
parent(tatiana, ludmila).
parent(vyacheslav, ruslan).
parent(tatiana, ruslan).

parent(alexandr, anton).
parent(ludmila, anton).
parent(alexandr, eduard).
parent(ludmila, eduard).
parent(alexandr, madlen).
parent(ludmila, madlen).

parent(anton, artem).
parent(yana, artem).

parent(ruslan, diana).
parent(ruslan, vika).

% События регистрации ЗАГС.
% -------------------------------------------------------------------
born(dmitriy, 2004).
born(denis, 2007).
born(angelina, 2018).
born(larisa, 1975).
born(andrew, 1980).
born(mama, 1981).
born(nikolay, 1950).
born(tamara, 1953).
born(georgiy, 1913).
born(agafya, 1915).
born(tatiana, 1941).
born(vyacheslav, 1948).
born(galina, 1943).
born(vladimir, 1939).
born(ludmila, 1977).
born(alexandr, 1975).
born(anton, 1996).
born(eduard, 2001).
born(madlen, 2007).

married(nikolay, tamara, 2005).
married(georgiy, agafya, 1930).
married(boris, galina, 1998).
married(andrew, mama, 2003).
married(vyacheslav, tatiana, 1980).
married(alexandr, ludmila, 1992).
married(anton, yana, 2023).

divorced(anton, yana, 2025).
divorced(georgiy, agafya, 1990).
divorced(andrew, mama, 2007).
divorced(alexandr, ludmila, 2018).

died(georgiy, 2006).
died(nikolay, 2023).
died(vladimir, 2010).
died(agafya, 2013).

% Правила.
% -------------------------------------------------------------------
