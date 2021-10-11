#ifndef BASICS_H
#define BASICS_H

int add(int a, int b) {
    return a + b;
}

class Numbers {
    const int _a, _b;

public:
    Numbers(): _a(0), _b(0) {}
    Numbers(int a, int b): _a(a), _b(b) {}

    int add() {
        return _a + _b;
    }

    int sub() {
        return _a - _b;
    }

    int mul() {
        return _a * _b;
    }

    float div() {
        return static_cast<float>(_a) / static_cast<float>(_b);
    }
};

#endif
