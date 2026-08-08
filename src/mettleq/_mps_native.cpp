#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <string>
#include <stdexcept>
#include <vector>

namespace {

using Pair = std::pair<int, int>;
using Complex = std::complex<float>;
using ComplexDouble = std::complex<double>;

struct ArrayView {
  PyArrayObject* object = nullptr;
  const Complex* data = nullptr;
  int dl = 0;
  int physical = 0;
  int dr = 0;
};

void release_array_view(ArrayView* view) {
  Py_XDECREF(view->object);
  view->object = nullptr;
  view->data = nullptr;
}

bool get_tensor(PyObject* object, ArrayView* view, const char* name) {
  PyObject* array = PyArray_FROM_OTF(
      object, NPY_COMPLEX64, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
  if (!array) return false;
  view->object = reinterpret_cast<PyArrayObject*>(array);
  if (PyArray_NDIM(view->object) != 3 || PyArray_DIM(view->object, 1) != 2) {
    PyErr_Format(PyExc_ValueError, "%s must have shape (Dl, 2, Dr)", name);
    release_array_view(view);
    return false;
  }
  const npy_intp dl = PyArray_DIM(view->object, 0);
  const npy_intp dr = PyArray_DIM(view->object, 2);
  if (dl < 1 || dr < 1 || dl > std::numeric_limits<int>::max() ||
      dr > std::numeric_limits<int>::max()) {
    PyErr_SetString(PyExc_ValueError, "MPS bond dimensions are out of range");
    release_array_view(view);
    return false;
  }
  view->dl = static_cast<int>(dl);
  view->physical = 2;
  view->dr = static_cast<int>(dr);
  view->data = static_cast<const Complex*>(PyArray_DATA(view->object));
  return true;
}

PyObject* make_array(const std::vector<Complex>& data,
                     const std::vector<npy_intp>& shape) {
  PyObject* result = PyArray_SimpleNew(
      static_cast<int>(shape.size()), const_cast<npy_intp*>(shape.data()),
      NPY_COMPLEX64);
  if (!result) return nullptr;
  std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(result)),
              data.data(), data.size() * sizeof(Complex));
  return result;
}

bool read_gate(PyObject* object, std::vector<Complex>* gate) {
  PyObject* array = PyArray_FROM_OTF(
      object, NPY_COMPLEX64, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
  if (!array) return false;
  PyArrayObject* matrix = reinterpret_cast<PyArrayObject*>(array);
  if (PyArray_NDIM(matrix) != 2 || PyArray_DIM(matrix, 0) != 4 ||
      PyArray_DIM(matrix, 1) != 4) {
    Py_DECREF(array);
    PyErr_SetString(PyExc_ValueError, "two-qubit gate must have shape (4, 4)");
    return false;
  }
  const auto* values = static_cast<const Complex*>(PyArray_DATA(matrix));
  gate->assign(values, values + 16);
  Py_DECREF(array);
  return true;
}

bool read_values(PyObject* object, int count, std::vector<Complex>* values) {
  PyObject* array = PyArray_FROM_OTF(
      object, NPY_COMPLEX64, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
  if (!array) return false;
  PyArrayObject* vector = reinterpret_cast<PyArrayObject*>(array);
  if (PyArray_SIZE(vector) != count) {
    Py_DECREF(array);
    PyErr_Format(PyExc_ValueError, "native gate payload must contain %d values", count);
    return false;
  }
  const auto* data = static_cast<const Complex*>(PyArray_DATA(vector));
  values->assign(data, data + count);
  Py_DECREF(array);
  return true;
}

struct NativeTensor {
  int dl = 0;
  int dr = 0;
  std::vector<ComplexDouble> data;
};

bool read_tensors(PyObject* object, std::vector<NativeTensor>* tensors) {
  PyObject* sequence = PySequence_Fast(object, "MPS tensors must be a sequence");
  if (!sequence) return false;
  const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence);
  if (count < 1) {
    Py_DECREF(sequence);
    PyErr_SetString(PyExc_ValueError, "an MPS needs at least one tensor");
    return false;
  }
  tensors->clear();
  tensors->reserve(static_cast<size_t>(count));
  for (Py_ssize_t index = 0; index < count; ++index) {
    PyObject* array = PyArray_FROM_OTF(
        PySequence_Fast_GET_ITEM(sequence, index), NPY_COMPLEX64,
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!array) {
      Py_DECREF(sequence);
      return false;
    }
    PyArrayObject* tensor = reinterpret_cast<PyArrayObject*>(array);
    if (PyArray_NDIM(tensor) != 3 || PyArray_DIM(tensor, 1) != 2) {
      Py_DECREF(array);
      Py_DECREF(sequence);
      PyErr_SetString(PyExc_ValueError, "MPS tensors must have shape (Dl, 2, Dr)");
      return false;
    }
    const npy_intp dl = PyArray_DIM(tensor, 0);
    const npy_intp dr = PyArray_DIM(tensor, 2);
    if (dl < 1 || dr < 1 || dl > std::numeric_limits<int>::max() ||
        dr > std::numeric_limits<int>::max()) {
      Py_DECREF(array);
      Py_DECREF(sequence);
      PyErr_SetString(PyExc_ValueError, "MPS tensor dimensions are out of range");
      return false;
    }
    NativeTensor value;
    value.dl = static_cast<int>(dl);
    value.dr = static_cast<int>(dr);
    const auto* data = static_cast<const Complex*>(PyArray_DATA(tensor));
    value.data.reserve(static_cast<size_t>(dl) * 2 * dr);
    for (npy_intp item = 0; item < dl * 2 * dr; ++item) {
      value.data.emplace_back(static_cast<double>(data[item].real()),
                              static_cast<double>(data[item].imag()));
    }
    Py_DECREF(array);
    if (!tensors->empty() && tensors->back().dr != value.dl) {
      Py_DECREF(sequence);
      PyErr_SetString(PyExc_ValueError, "MPS tensor bonds do not match");
      return false;
    }
    tensors->push_back(std::move(value));
  }
  Py_DECREF(sequence);
  return true;
}

bool read_ints(PyObject* object, std::vector<int>* values,
               const char* description) {
  PyObject* sequence = PySequence_Fast(object, description);
  if (!sequence) return false;
  values->clear();
  const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence);
  values->reserve(static_cast<size_t>(count));
  for (Py_ssize_t index = 0; index < count; ++index) {
    const long value = PyLong_AsLong(PySequence_Fast_GET_ITEM(sequence, index));
    if (PyErr_Occurred()) {
      Py_DECREF(sequence);
      return false;
    }
    values->push_back(static_cast<int>(value));
  }
  Py_DECREF(sequence);
  return true;
}

bool validate_mapping(const std::vector<int>& mapping, int n) {
  if (static_cast<int>(mapping.size()) != n) {
    PyErr_SetString(PyExc_ValueError, "logical-to-site mapping has wrong length");
    return false;
  }
  std::vector<bool> seen(static_cast<size_t>(n), false);
  for (int site : mapping) {
    if (site < 0 || site >= n || seen[site]) {
      PyErr_SetString(PyExc_ValueError, "logical-to-site mapping is not a permutation");
      return false;
    }
    seen[site] = true;
  }
  return true;
}

ComplexDouble mps_norm_squared(const std::vector<NativeTensor>& tensors) {
  std::vector<ComplexDouble> environment(1, ComplexDouble(1.0, 0.0));
  for (const auto& tensor : tensors) {
    const int dl = tensor.dl;
    const int dr = tensor.dr;
    std::vector<ComplexDouble> next(static_cast<size_t>(dr) * dr,
                                    ComplexDouble(0.0, 0.0));
    const int old_dim = static_cast<int>(std::sqrt(environment.size()));
    for (int a = 0; a < old_dim; ++a) {
      for (int b = 0; b < old_dim; ++b) {
        const ComplexDouble env = environment[a * old_dim + b];
        for (int r = 0; r < dr; ++r) {
          for (int q = 0; q < dr; ++q) {
            ComplexDouble value(0.0, 0.0);
            for (int physical = 0; physical < 2; ++physical) {
              const auto ket = tensor.data[(a * 2 + physical) * dr + r];
              const auto bra = std::conj(tensor.data[(b * 2 + physical) * dr + q]);
              value += ket * bra;
            }
            next[r * dr + q] += env * value;
          }
        }
      }
    }
    environment.swap(next);
  }
  return environment.empty() ? ComplexDouble(0.0, 0.0) : environment[0];
}

ComplexDouble transfer_operator(const std::vector<NativeTensor>& tensors,
                                const std::vector<int>& physical_sites,
                                const std::vector<ComplexDouble>& operators,
                                const std::vector<int>& operator_sites) {
  std::vector<ComplexDouble> environment(1, ComplexDouble(1.0, 0.0));
  for (size_t site = 0; site < tensors.size(); ++site) {
    const auto& tensor = tensors[site];
    const int dl = tensor.dl;
    const int dr = tensor.dr;
    std::vector<ComplexDouble> next(static_cast<size_t>(dr) * dr,
                                    ComplexDouble(0.0, 0.0));
    const int old_dim = static_cast<int>(std::sqrt(environment.size()));
    int operator_index = -1;
    for (size_t index = 0; index < physical_sites.size(); ++index) {
      if (physical_sites[index] == static_cast<int>(site)) {
        operator_index = static_cast<int>(index);
        break;
      }
    }
#ifdef __APPLE__
    std::vector<ComplexDouble> ket_matrices[2];
    std::vector<ComplexDouble> bra_matrices[2];
    for (int physical = 0; physical < 2; ++physical) {
      ket_matrices[physical].resize(static_cast<size_t>(old_dim) * dr);
      bra_matrices[physical].resize(static_cast<size_t>(old_dim) * dr);
      for (int a = 0; a < old_dim; ++a) {
        for (int r = 0; r < dr; ++r) {
          const auto value = tensor.data[(a * 2 + physical) * dr + r];
          ket_matrices[physical][a * dr + r] = value;
          bra_matrices[physical][a * dr + r] = std::conj(value);
        }
      }
    }
    const ComplexDouble one(1.0, 0.0);
    const ComplexDouble zero(0.0, 0.0);
    for (int ket_bit = 0; ket_bit < 2; ++ket_bit) {
      for (int bra_bit = 0; bra_bit < 2; ++bra_bit) {
        ComplexDouble coefficient =
            ket_bit == bra_bit ? one : zero;
        if (operator_index >= 0) {
          const size_t offset = static_cast<size_t>(operator_index) * 4;
          coefficient = operators[offset + bra_bit * 2 + ket_bit];
        }
        if (std::abs(coefficient) == 0.0) continue;
        std::vector<ComplexDouble> left_product(static_cast<size_t>(dr) * old_dim,
                                                zero);
        cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dr, old_dim,
                    old_dim, &one, ket_matrices[ket_bit].data(), dr,
                    environment.data(), old_dim, &zero, left_product.data(),
                    old_dim);
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dr, dr,
                    old_dim, &coefficient, left_product.data(), old_dim,
                    bra_matrices[bra_bit].data(), dr, &one, next.data(), dr);
      }
    }
    environment.swap(next);
    continue;
#endif
    for (int a = 0; a < old_dim; ++a) {
      for (int b = 0; b < old_dim; ++b) {
        const ComplexDouble env = environment[a * old_dim + b];
        for (int r = 0; r < dr; ++r) {
          for (int q = 0; q < dr; ++q) {
            ComplexDouble value(0.0, 0.0);
            for (int ket_bit = 0; ket_bit < 2; ++ket_bit) {
              for (int bra_bit = 0; bra_bit < 2; ++bra_bit) {
                ComplexDouble op = ket_bit == bra_bit
                                       ? ComplexDouble(1.0, 0.0)
                                       : ComplexDouble(0.0, 0.0);
                if (operator_index >= 0) {
                  const size_t offset = static_cast<size_t>(operator_index) * 4;
                  op = operators[offset + bra_bit * 2 + ket_bit];
                }
                value += tensor.data[(a * 2 + ket_bit) * dr + r] * op *
                         std::conj(tensor.data[(b * 2 + bra_bit) * dr + q]);
              }
            }
            next[r * dr + q] += env * value;
          }
        }
      }
    }
    environment.swap(next);
  }
  return environment.empty() ? ComplexDouble(0.0, 0.0) : environment[0];
}

PyObject* native_expectation_product(PyObject*, PyObject* args) {
  PyObject* tensors_object = nullptr;
  PyObject* wires_object = nullptr;
  PyObject* operators_object = nullptr;
  PyObject* mapping_object = nullptr;
  double supplied_norm = 0.0;
  if (!PyArg_ParseTuple(args, "OOOO|d", &tensors_object, &wires_object,
                        &operators_object, &mapping_object, &supplied_norm)) {
    return nullptr;
  }
  std::vector<NativeTensor> tensors;
  std::vector<int> wires;
  std::vector<int> mapping;
  if (!read_tensors(tensors_object, &tensors) ||
      !read_ints(wires_object, &wires, "observable wires must be a sequence") ||
      !read_ints(mapping_object, &mapping, "mapping must be a sequence")) {
    return nullptr;
  }
  const int n = static_cast<int>(tensors.size());
  if (!validate_mapping(mapping, n)) return nullptr;
  if (wires.empty()) return PyComplex_FromDoubles(1.0, 0.0);
  PyObject* array = PyArray_FROM_OTF(
      operators_object, NPY_COMPLEX128,
      NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
  if (!array) return nullptr;
  PyArrayObject* operators = reinterpret_cast<PyArrayObject*>(array);
  if (PyArray_NDIM(operators) != 3 || PyArray_DIM(operators, 0) !=
      static_cast<npy_intp>(wires.size()) || PyArray_DIM(operators, 1) != 2 ||
      PyArray_DIM(operators, 2) != 2) {
    Py_DECREF(array);
    PyErr_SetString(PyExc_ValueError, "native observable operators must have shape (k, 2, 2)");
    return nullptr;
  }
  const auto* data = static_cast<const ComplexDouble*>(PyArray_DATA(operators));
  std::vector<int> physical_sites;
  physical_sites.reserve(wires.size());
  for (int wire : wires) {
    if (wire < 0 || wire >= n) {
      Py_DECREF(array);
      PyErr_SetString(PyExc_ValueError, "observable wire is out of range");
      return nullptr;
    }
    physical_sites.push_back(mapping[wire]);
  }
  std::vector<ComplexDouble> operator_values(
      data, data + static_cast<size_t>(wires.size()) * 4);
  Py_DECREF(array);
  const ComplexDouble numerator = transfer_operator(
      tensors, physical_sites, operator_values, physical_sites);
  const ComplexDouble denominator = supplied_norm > 0.0
                                        ? ComplexDouble(supplied_norm * supplied_norm, 0.0)
                                        : mps_norm_squared(tensors);
  if (!(std::abs(denominator) > std::numeric_limits<double>::min())) {
    PyErr_SetString(PyExc_RuntimeError, "native MPS has zero norm");
    return nullptr;
  }
  return PyComplex_FromDoubles(
      (numerator / denominator).real(), (numerator / denominator).imag());
}

PyObject* native_probabilities(PyObject*, PyObject* args) {
  PyObject* tensors_object = nullptr;
  PyObject* wires_object = nullptr;
  PyObject* mapping_object = nullptr;
  double supplied_norm = 0.0;
  if (!PyArg_ParseTuple(args, "OOO|d", &tensors_object, &wires_object,
                        &mapping_object, &supplied_norm)) {
    return nullptr;
  }
  std::vector<NativeTensor> tensors;
  std::vector<int> wires;
  std::vector<int> mapping;
  if (!read_tensors(tensors_object, &tensors) ||
      !read_ints(wires_object, &wires, "probability wires must be a sequence") ||
      !read_ints(mapping_object, &mapping, "mapping must be a sequence")) {
    return nullptr;
  }
  const int n = static_cast<int>(tensors.size());
  if (!validate_mapping(mapping, n)) return nullptr;
  for (int wire : wires) {
    if (wire < 0 || wire >= n) {
      PyErr_SetString(PyExc_ValueError, "probability wire is out of range");
      return nullptr;
    }
  }
  if (wires.size() >= sizeof(size_t) * 8 || (size_t(1) << wires.size()) >
      static_cast<size_t>(std::numeric_limits<npy_intp>::max())) {
    PyErr_SetString(PyExc_ValueError, "probability request is too large");
    return nullptr;
  }
  const size_t outcomes = size_t(1) << wires.size();
  const double norm = supplied_norm > 0.0
                          ? supplied_norm * supplied_norm
                          : std::real(mps_norm_squared(tensors));
  if (!(norm > std::numeric_limits<double>::min())) {
    PyErr_SetString(PyExc_RuntimeError, "native MPS has zero norm");
    return nullptr;
  }
  std::vector<ComplexDouble> identity(4, ComplexDouble(0.0, 0.0));
  identity[0] = identity[3] = ComplexDouble(1.0, 0.0);
  std::vector<ComplexDouble> projector(4, ComplexDouble(0.0, 0.0));
  std::vector<int> physical_sites;
  std::vector<ComplexDouble> operators;
  std::vector<double> probabilities(outcomes, 0.0);
  for (size_t outcome = 0; outcome < outcomes; ++outcome) {
    physical_sites.clear();
    operators.clear();
    for (size_t index = 0; index < wires.size(); ++index) {
      physical_sites.push_back(mapping[wires[index]]);
      projector.assign(4, ComplexDouble(0.0, 0.0));
      const int bit = static_cast<int>((outcome >> (wires.size() - 1 - index)) & 1);
      projector[bit * 2 + bit] = ComplexDouble(1.0, 0.0);
      operators.insert(operators.end(), projector.begin(), projector.end());
    }
    const auto value = transfer_operator(tensors, physical_sites, operators,
                                         physical_sites);
    probabilities[outcome] = std::max(0.0, value.real() / norm);
  }
  double total = 0.0;
  for (double value : probabilities) total += value;
  if (!(total > 0.0)) {
    PyErr_SetString(PyExc_RuntimeError, "native MPS marginal has zero probability");
    return nullptr;
  }
  npy_intp shape = static_cast<npy_intp>(outcomes);
  PyObject* result = PyArray_SimpleNew(1, &shape, NPY_FLOAT64);
  if (!result) return nullptr;
  auto* output = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(result)));
  for (size_t index = 0; index < outcomes; ++index) output[index] = probabilities[index] / total;
  return result;
}

PyObject* native_sample(PyObject*, PyObject* args) {
  PyObject* tensors_object = nullptr;
  PyObject* wires_object = nullptr;
  PyObject* mapping_object = nullptr;
  PyObject* random_object = nullptr;
  if (!PyArg_ParseTuple(args, "OOOO", &tensors_object, &wires_object,
                        &mapping_object, &random_object)) {
    return nullptr;
  }
  std::vector<NativeTensor> tensors;
  std::vector<int> wires;
  std::vector<int> mapping;
  if (!read_tensors(tensors_object, &tensors) ||
      !read_ints(wires_object, &wires, "sample wires must be a sequence") ||
      !read_ints(mapping_object, &mapping, "mapping must be a sequence")) {
    return nullptr;
  }
  const int n = static_cast<int>(tensors.size());
  if (!validate_mapping(mapping, n)) return nullptr;
  for (int wire : wires) {
    if (wire < 0 || wire >= n) {
      PyErr_SetString(PyExc_ValueError, "sample wire is out of range");
      return nullptr;
    }
  }
  PyObject* random_array_object = PyArray_FROM_OTF(
      random_object, NPY_FLOAT64, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
  if (!random_array_object) return nullptr;
  PyArrayObject* random_array = reinterpret_cast<PyArrayObject*>(random_array_object);
  if (PyArray_NDIM(random_array) != 2 || PyArray_DIM(random_array, 1) != n) {
    Py_DECREF(random_array_object);
    PyErr_SetString(PyExc_ValueError, "native sample random stream has wrong shape");
    return nullptr;
  }
  const int shots = static_cast<int>(PyArray_DIM(random_array, 0));
  const auto* random_values = static_cast<const double*>(PyArray_DATA(random_array));
  std::vector<std::vector<ComplexDouble>> right_env(
      static_cast<size_t>(n + 1));
  right_env[n] = std::vector<ComplexDouble>(1, ComplexDouble(1.0, 0.0));
  for (int site = n - 1; site >= 0; --site) {
    const auto& tensor = tensors[site];
    const int dl = tensor.dl;
    const int dr = tensor.dr;
    right_env[site].assign(static_cast<size_t>(dl) * dl,
                           ComplexDouble(0.0, 0.0));
    for (int a = 0; a < dl; ++a) {
      for (int b = 0; b < dl; ++b) {
        ComplexDouble value(0.0, 0.0);
        for (int physical = 0; physical < 2; ++physical) {
          for (int r = 0; r < dr; ++r) {
            for (int q = 0; q < dr; ++q) {
              value += tensor.data[(a * 2 + physical) * dr + r] *
                       std::conj(tensor.data[(b * 2 + physical) * dr + q]) *
                       right_env[site + 1][r * dr + q];
            }
          }
        }
        right_env[site][a * dl + b] = value;
      }
    }
  }
  npy_intp shape[2] = {shots, static_cast<npy_intp>(wires.size())};
  PyObject* result = PyArray_SimpleNew(2, shape, NPY_INT64);
  if (!result) {
    Py_DECREF(random_array_object);
    return nullptr;
  }
  auto* output = static_cast<int64_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(result)));
  const double tiny = std::numeric_limits<double>::min();
  for (int shot = 0; shot < shots; ++shot) {
    std::vector<ComplexDouble> left_env(1, ComplexDouble(1.0, 0.0));
    std::vector<int> physical_bits(n, 0);
    for (int site = 0; site < n; ++site) {
      const auto& tensor = tensors[site];
      const int dl = tensor.dl;
      const int dr = tensor.dr;
      const int old_dim = static_cast<int>(std::sqrt(left_env.size()));
      double weights[2] = {0.0, 0.0};
      std::vector<std::vector<ComplexDouble>> updates(2);
      for (int bit = 0; bit < 2; ++bit) {
        updates[bit].assign(static_cast<size_t>(dr) * dr,
                            ComplexDouble(0.0, 0.0));
        for (int a = 0; a < old_dim; ++a) {
          for (int b = 0; b < old_dim; ++b) {
            const auto env = left_env[a * old_dim + b];
            for (int r = 0; r < dr; ++r) {
              for (int q = 0; q < dr; ++q) {
                updates[bit][r * dr + q] +=
                    env * tensor.data[(a * 2 + bit) * dr + r] *
                    std::conj(tensor.data[(b * 2 + bit) * dr + q]);
              }
            }
          }
        }
        for (int r = 0; r < dr; ++r) {
          for (int q = 0; q < dr; ++q) {
            weights[bit] +=
                (updates[bit][r * dr + q] * right_env[site + 1][r * dr + q]).real();
          }
        }
        weights[bit] = std::max(0.0, weights[bit]);
      }
      const double total = weights[0] + weights[1];
      if (!(total > tiny)) {
        Py_DECREF(random_array_object);
        Py_DECREF(result);
        PyErr_SetString(PyExc_RuntimeError, "native MPS conditional probability is zero");
        return nullptr;
      }
      const int bit = random_values[static_cast<size_t>(shot) * n + site] <
                              weights[1] / total
                          ? 1
                          : 0;
      physical_bits[site] = bit;
      left_env = updates[bit];
      for (auto& value : left_env) value /= std::max(weights[bit], tiny);
    }
    for (size_t index = 0; index < wires.size(); ++index) {
      output[static_cast<size_t>(shot) * wires.size() + index] =
          physical_bits[mapping[wires[index]]];
    }
  }
  Py_DECREF(random_array_object);
  return result;
}

PyObject* native_statevector(PyObject*, PyObject* args) {
  PyObject* tensors_object = nullptr;
  PyObject* mapping_object = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &tensors_object, &mapping_object)) return nullptr;
  std::vector<NativeTensor> tensors;
  std::vector<int> mapping;
  if (!read_tensors(tensors_object, &tensors) ||
      !read_ints(mapping_object, &mapping, "mapping must be a sequence")) return nullptr;
  const int n = static_cast<int>(tensors.size());
  if (!validate_mapping(mapping, n)) return nullptr;
  std::vector<ComplexDouble> amplitudes(1, ComplexDouble(1.0, 0.0));
  int bond = 1;
  for (const auto& tensor : tensors) {
    std::vector<ComplexDouble> next(amplitudes.size() * 2 * tensor.dr,
                                    ComplexDouble(0.0, 0.0));
    const size_t prefixes = amplitudes.size() / static_cast<size_t>(bond);
    for (size_t prefix = 0; prefix < prefixes; ++prefix) {
      for (int left = 0; left < bond; ++left) {
        const auto amplitude = amplitudes[prefix * bond + left];
        for (int physical = 0; physical < 2; ++physical) {
          for (int right = 0; right < tensor.dr; ++right) {
            const size_t index = (prefix * 2 + physical) * tensor.dr + right;
            next[index] += amplitude * tensor.data[(left * 2 + physical) * tensor.dr + right];
          }
        }
      }
    }
    amplitudes.swap(next);
    bond = tensor.dr;
  }
  const size_t dimension = size_t(1) << n;
  std::vector<Complex> logical(dimension, Complex(0.0f, 0.0f));
  for (size_t physical_index = 0; physical_index < dimension; ++physical_index) {
    size_t logical_index = 0;
    for (int logical_wire = 0; logical_wire < n; ++logical_wire) {
      const int physical_site = mapping[logical_wire];
      const size_t bit = (physical_index >> (n - 1 - physical_site)) & 1;
      logical_index |= bit << (n - 1 - logical_wire);
    }
    logical[logical_index] = Complex(
        static_cast<float>(amplitudes[physical_index].real()),
        static_cast<float>(amplitudes[physical_index].imag()));
  }
  double norm = 0.0;
  for (const auto& value : logical) norm += std::norm(value);
  if (!(norm > std::numeric_limits<double>::min())) {
    PyErr_SetString(PyExc_RuntimeError, "native MPS produced a zero statevector");
    return nullptr;
  }
  const float inverse_norm = static_cast<float>(1.0 / std::sqrt(norm));
  for (auto& value : logical) value *= inverse_norm;
  return make_array(logical, {static_cast<npy_intp>(dimension)});
}

void merge_two_site(const ArrayView& left, const ArrayView& right,
                    std::vector<Complex>* tensor) {
  const int bond = left.dr;
  const int dl = left.dl;
  const int dr = right.dr;
  tensor->assign(static_cast<size_t>(dl) * 4 * dr, Complex(0.0f, 0.0f));
#ifdef __APPLE__
  const Complex alpha(1.0f, 0.0f);
  const Complex beta(0.0f, 0.0f);
  // A and B are already C-contiguous row-major matrices with the physical
  // index fused into the adjacent matrix dimension.
  cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dl * 2, dr * 2,
              bond, &alpha, left.data, bond, right.data, dr * 2, &beta,
              tensor->data(), dr * 2);
#else
  for (int l = 0; l < dl; ++l) {
    for (int p = 0; p < 2; ++p) {
      for (int q = 0; q < 2; ++q) {
        for (int r = 0; r < dr; ++r) {
          Complex value(0.0f, 0.0f);
          for (int k = 0; k < bond; ++k) {
            value += left.data[(l * 2 + p) * bond + k] *
                     right.data[(k * 2 + q) * dr + r];
          }
          (*tensor)[((l * 2 + p) * 2 + q) * dr + r] = value;
        }
      }
    }
  }
#endif
}

void apply_payload(std::vector<Complex>* tensor, int dl, int dr,
                   const std::vector<Complex>& payload,
                   const std::string& kind) {
  if (kind == "generic") {
    std::vector<Complex> transformed(tensor->size(), Complex(0.0f, 0.0f));
    for (int l = 0; l < dl; ++l) {
      for (int q = 0; q < dr; ++q) {
        for (int out = 0; out < 4; ++out) {
          Complex value(0.0f, 0.0f);
          for (int in = 0; in < 4; ++in) {
            const int p = in / 2;
            const int r = in % 2;
            value += payload[out * 4 + in] *
                     (*tensor)[((l * 2 + p) * 2 + r) * dr + q];
          }
          const int p = out / 2;
          const int r = out % 2;
          transformed[((l * 2 + p) * 2 + r) * dr + q] = value;
        }
      }
    }
    tensor->swap(transformed);
    return;
  }
  std::vector<Complex> transformed(tensor->size(), Complex(0.0f, 0.0f));
  for (int l = 0; l < dl; ++l) {
    for (int p = 0; p < 2; ++p) {
      for (int q = 0; q < 2; ++q) {
        const int input = p * 2 + q;
        int output = input;
        if (kind == "swap") output = q * 2 + p;
        if (kind == "cnot") output = p * 2 + (q ^ p);
        if (kind == "cnot_reverse") output = (p ^ q) * 2 + q;
        const Complex factor = kind == "diagonal" || kind == "zz"
                                   ? payload[input]
                                   : Complex(1.0f, 0.0f);
        const int out_p = output / 2;
        const int out_q = output % 2;
        for (int r = 0; r < dr; ++r) {
          transformed[((l * 2 + out_p) * 2 + out_q) * dr + r] =
              (*tensor)[((l * 2 + p) * 2 + q) * dr + r] * factor;
        }
      }
    }
  }
  tensor->swap(transformed);
}

struct SplitResult {
  std::vector<Complex> left;
  std::vector<Complex> right;
  int rank_before = 0;
  int rank_kept = 0;
  float discarded_weight = 0.0f;
  float total_weight = 0.0f;
  float kept_weight = 0.0f;
};

#ifdef __APPLE__
bool accelerate_svd(const std::vector<Complex>& row_major, int rows, int cols,
                    std::vector<Complex>* u_row_major,
                    std::vector<float>* singular_values,
                    std::vector<Complex>* vh_row_major) {
  const int rank = std::min(rows, cols);
  std::vector<Complex> matrix(static_cast<size_t>(rows) * cols);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      matrix[col * rows + row] = row_major[row * cols + col];
    }
  }
  std::vector<Complex> u(static_cast<size_t>(rows) * rank);
  std::vector<Complex> vt(static_cast<size_t>(rank) * cols);
  singular_values->assign(rank, 0.0f);
  std::vector<float> rwork(static_cast<size_t>(5 * rank * rank + 7 * rank + 1));
  std::vector<__LAPACK_int> iwork(static_cast<size_t>(8 * rank + 1));
  __LAPACK_int m = rows;
  __LAPACK_int n = cols;
  __LAPACK_int lda = std::max(1, rows);
  __LAPACK_int ldu = std::max(1, rows);
  __LAPACK_int ldvt = std::max(1, rank);
  __LAPACK_int lwork = -1;
  __LAPACK_int info = 0;
  char jobz = 'S';
  Complex work_query(0.0f, 0.0f);
  cgesdd_(&jobz, &m, &n, matrix.data(), &lda, singular_values->data(),
          u.data(), &ldu, vt.data(), &ldvt, &work_query, &lwork,
          rwork.data(), iwork.data(), &info);
  if (info != 0 || !std::isfinite(work_query.real())) return false;
  lwork = std::max<__LAPACK_int>(1, static_cast<__LAPACK_int>(
      std::ceil(static_cast<double>(work_query.real()))));
  std::vector<Complex> work(static_cast<size_t>(lwork));
  cgesdd_(&jobz, &m, &n, matrix.data(), &lda, singular_values->data(),
          u.data(), &ldu, vt.data(), &ldvt, work.data(), &lwork,
          rwork.data(), iwork.data(), &info);
  if (info != 0) return false;
  u_row_major->assign(static_cast<size_t>(rows) * rank, Complex(0.0f, 0.0f));
  vh_row_major->assign(static_cast<size_t>(rank) * cols, Complex(0.0f, 0.0f));
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < rank; ++col) {
      (*u_row_major)[row * rank + col] = u[col * rows + row];
    }
  }
  for (int row = 0; row < rank; ++row) {
    for (int col = 0; col < cols; ++col) {
      (*vh_row_major)[row * cols + col] = vt[col * rank + row];
    }
  }
  return true;
}
#endif

bool split_tensor(const std::vector<Complex>& tensor, int dl, int dr, int dmax,
                  float eps, bool renormalize, SplitResult* result) {
  const int rows = dl * 2;
  const int cols = dr * 2;
  std::vector<Complex> matrix(static_cast<size_t>(rows) * cols);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      matrix[row * cols + col] = tensor[row * cols + col];
    }
  }
  std::vector<Complex> u;
  std::vector<Complex> vh;
  std::vector<float> singular_values;
#ifdef __APPLE__
  if (!accelerate_svd(matrix, rows, cols, &u, &singular_values, &vh)) {
    PyErr_SetString(PyExc_RuntimeError, "Accelerate complex SVD failed");
    return false;
  }
#else
  PyErr_SetString(PyExc_RuntimeError, "native MPS SVD requires Apple Accelerate");
  return false;
#endif
  const int rank = static_cast<int>(singular_values.size());
  int rank_eps = rank;
  if (rank > 0 && eps > 0.0f) {
    const float threshold = eps * singular_values[0];
    rank_eps = 0;
    while (rank_eps < rank && singular_values[rank_eps] >= threshold) ++rank_eps;
  }
  const int keep = std::max(1, std::min({dmax, rank, rank_eps}));
  result->rank_before = rank;
  result->rank_kept = keep;
  result->left.assign(static_cast<size_t>(dl) * 2 * keep, Complex(0.0f, 0.0f));
  result->right.assign(static_cast<size_t>(keep) * 2 * dr, Complex(0.0f, 0.0f));
  for (int i = 0; i < rank; ++i) {
    const float value = singular_values[i];
    result->total_weight += value * value;
    if (i < keep) result->kept_weight += value * value;
    else result->discarded_weight += value * value;
  }
  if (!(result->kept_weight > std::numeric_limits<float>::min()) ||
      !std::isfinite(result->kept_weight)) {
    PyErr_SetString(PyExc_RuntimeError, "native SVD retained an invalid spectrum");
    return false;
  }
  const float normalization = renormalize ? std::sqrt(result->kept_weight) : 1.0f;
  for (int row = 0; row < dl * 2; ++row) {
    for (int col = 0; col < keep; ++col) {
      result->left[row * keep + col] = u[row * rank + col];
    }
  }
  for (int row = 0; row < keep; ++row) {
    for (int col = 0; col < 2 * dr; ++col) {
      result->right[row * 2 * dr + col] =
          vh[row * cols + col] * (singular_values[row] / normalization);
    }
  }
  return true;
}

#ifdef __APPLE__
bool accelerate_qr(const std::vector<Complex>& row_major, int rows, int cols,
                  std::vector<Complex>* q_row_major,
                  std::vector<Complex>* r_row_major, int* rank) {
  const int reduced_rank = std::min(rows, cols);
  std::vector<Complex> matrix(static_cast<size_t>(rows) * cols);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      matrix[col * rows + row] = row_major[row * cols + col];
    }
  }
  std::vector<Complex> tau(reduced_rank);
  __LAPACK_int m = rows;
  __LAPACK_int n = cols;
  __LAPACK_int lda = std::max(1, rows);
  __LAPACK_int info = 0;
  __LAPACK_int lwork = -1;
  Complex query(0.0f, 0.0f);
  cgeqrf_(&m, &n, matrix.data(), &lda, tau.data(), &query, &lwork, &info);
  if (info != 0 || !std::isfinite(query.real())) return false;
  lwork = std::max<__LAPACK_int>(1, static_cast<__LAPACK_int>(
      std::ceil(static_cast<double>(query.real()))));
  std::vector<Complex> work(static_cast<size_t>(lwork));
  cgeqrf_(&m, &n, matrix.data(), &lda, tau.data(), work.data(), &lwork, &info);
  if (info != 0) return false;
  r_row_major->assign(static_cast<size_t>(reduced_rank) * cols,
                      Complex(0.0f, 0.0f));
  for (int row = 0; row < reduced_rank; ++row) {
    for (int col = row; col < cols; ++col) {
      (*r_row_major)[row * cols + col] = matrix[col * rows + row];
    }
  }
  __LAPACK_int qn = reduced_rank;
  __LAPACK_int qk = reduced_rank;
  lwork = -1;
  query = Complex(0.0f, 0.0f);
  cungqr_(&m, &qn, &qk, matrix.data(), &lda, tau.data(), &query, &lwork,
          &info);
  if (info != 0 || !std::isfinite(query.real())) return false;
  lwork = std::max<__LAPACK_int>(1, static_cast<__LAPACK_int>(
      std::ceil(static_cast<double>(query.real()))));
  work.assign(static_cast<size_t>(lwork), Complex(0.0f, 0.0f));
  cungqr_(&m, &qn, &qk, matrix.data(), &lda, tau.data(), work.data(), &lwork,
          &info);
  if (info != 0) return false;
  q_row_major->assign(static_cast<size_t>(rows) * reduced_rank,
                      Complex(0.0f, 0.0f));
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < reduced_rank; ++col) {
      (*q_row_major)[row * reduced_rank + col] = matrix[col * rows + row];
    }
  }
  *rank = reduced_rank;
  return true;
}
#endif

PyObject* native_qr_move_right(PyObject*, PyObject* args) {
  PyObject* left_object = nullptr;
  PyObject* right_object = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &left_object, &right_object)) return nullptr;
  ArrayView left;
  ArrayView right;
  if (!get_tensor(left_object, &left, "left tensor") ||
      !get_tensor(right_object, &right, "right tensor")) {
    release_array_view(&left);
    release_array_view(&right);
    return nullptr;
  }
  if (left.dr != right.dl) {
    release_array_view(&left);
    release_array_view(&right);
    PyErr_SetString(PyExc_ValueError, "native QR tensors have incompatible bonds");
    return nullptr;
  }
  const int rows = left.dl * 2;
  const int cols = left.dr;
  std::vector<Complex> matrix(static_cast<size_t>(rows) * cols);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      matrix[row * cols + col] = left.data[row * cols + col];
    }
  }
  std::vector<Complex> q;
  std::vector<Complex> r;
  int rank = 0;
#ifdef __APPLE__
  const bool ok = accelerate_qr(matrix, rows, cols, &q, &r, &rank);
#else
  const bool ok = false;
#endif
  if (!ok) {
    release_array_view(&left);
    release_array_view(&right);
    PyErr_SetString(PyExc_RuntimeError, "Accelerate complex QR failed");
    return nullptr;
  }
  const int right_dr = right.dr;
  std::vector<Complex> absorbed(static_cast<size_t>(rank) * 2 * right_dr,
                                Complex(0.0f, 0.0f));
#ifdef __APPLE__
  const Complex alpha(1.0f, 0.0f);
  const Complex beta(0.0f, 0.0f);
  cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rank, right_dr * 2,
              cols, &alpha, r.data(), cols, right.data, right_dr * 2, &beta,
              absorbed.data(), right_dr * 2);
#endif
  release_array_view(&left);
  release_array_view(&right);
  PyObject* q_array = make_array(q, {left.dl, 2, rank});
  PyObject* right_array = make_array(absorbed, {rank, 2, right_dr});
  if (!q_array || !right_array) {
    Py_XDECREF(q_array);
    Py_XDECREF(right_array);
    return nullptr;
  }
  PyObject* result = PyTuple_New(2);
  if (!result) {
    Py_DECREF(q_array);
    Py_DECREF(right_array);
    return nullptr;
  }
  PyTuple_SET_ITEM(result, 0, q_array);
  PyTuple_SET_ITEM(result, 1, right_array);
  return result;
}

PyObject* native_qr_move_left(PyObject*, PyObject* args) {
  PyObject* previous_object = nullptr;
  PyObject* current_object = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &previous_object, &current_object)) return nullptr;
  ArrayView previous;
  ArrayView current;
  if (!get_tensor(previous_object, &previous, "previous tensor") ||
      !get_tensor(current_object, &current, "current tensor")) {
    release_array_view(&previous);
    release_array_view(&current);
    return nullptr;
  }
  if (previous.dr != current.dl) {
    release_array_view(&previous);
    release_array_view(&current);
    PyErr_SetString(PyExc_ValueError, "native QR tensors have incompatible bonds");
    return nullptr;
  }
  const int rows = current.dr * 2;
  const int cols = current.dl;
  std::vector<Complex> matrix(static_cast<size_t>(rows) * cols);
  for (int physical = 0; physical < 2; ++physical) {
    for (int right = 0; right < current.dr; ++right) {
      const int row = physical * current.dr + right;
      for (int left = 0; left < current.dl; ++left) {
        matrix[row * cols + left] =
            current.data[(left * 2 + physical) * current.dr + right];
      }
    }
  }
  std::vector<Complex> q;
  std::vector<Complex> r;
  int rank = 0;
#ifdef __APPLE__
  const bool ok = accelerate_qr(matrix, rows, cols, &q, &r, &rank);
#else
  const bool ok = false;
#endif
  if (!ok) {
    release_array_view(&previous);
    release_array_view(&current);
    PyErr_SetString(PyExc_RuntimeError, "Accelerate complex QR failed");
    return nullptr;
  }
  const int previous_dl = previous.dl;
  std::vector<Complex> absorbed(static_cast<size_t>(previous_dl) * 2 * rank,
                                Complex(0.0f, 0.0f));
  std::vector<Complex> previous_matrix(static_cast<size_t>(previous_dl) * 2 * previous.dr);
  for (int left = 0; left < previous_dl; ++left) {
    for (int physical = 0; physical < 2; ++physical) {
      for (int bond = 0; bond < previous.dr; ++bond) {
        previous_matrix[(left * 2 + physical) * previous.dr + bond] =
            previous.data[(left * 2 + physical) * previous.dr + bond];
      }
    }
  }
#ifdef __APPLE__
  const Complex alpha(1.0f, 0.0f);
  const Complex beta(0.0f, 0.0f);
  cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasTrans, previous_dl * 2,
              rank, current.dl, &alpha, previous_matrix.data(), current.dl,
              r.data(), cols, &beta, absorbed.data(), rank);
#endif
  std::vector<Complex> q_transposed(static_cast<size_t>(rank) * 2 * current.dr,
                                    Complex(0.0f, 0.0f));
  for (int new_left = 0; new_left < rank; ++new_left) {
    for (int physical = 0; physical < 2; ++physical) {
      for (int right = 0; right < current.dr; ++right) {
        const int q_row = physical * current.dr + right;
        q_transposed[(new_left * 2 + physical) * current.dr + right] =
            q[q_row * rank + new_left];
      }
    }
  }
  const int current_dr = current.dr;
  release_array_view(&previous);
  release_array_view(&current);
  PyObject* previous_array = make_array(absorbed, {previous_dl, 2, rank});
  PyObject* current_array = make_array(q_transposed, {rank, 2, current_dr});
  if (!previous_array || !current_array) {
    Py_XDECREF(previous_array);
    Py_XDECREF(current_array);
    return nullptr;
  }
  PyObject* result = PyTuple_New(2);
  if (!result) {
    Py_DECREF(previous_array);
    Py_DECREF(current_array);
    return nullptr;
  }
  PyTuple_SET_ITEM(result, 0, previous_array);
  PyTuple_SET_ITEM(result, 1, current_array);
  return result;
}

PyObject* native_two_site_update(PyObject*, PyObject* args) {
  PyObject* left_object = nullptr;
  PyObject* right_object = nullptr;
  PyObject* payload_object = nullptr;
  PyObject* kind_object = nullptr;
  int dmax = 0;
  double eps = 0.0;
  int renormalize = 1;
  if (!PyArg_ParseTuple(args, "OOOOidp", &left_object, &right_object,
                        &payload_object, &kind_object, &dmax, &eps,
                        &renormalize)) {
    return nullptr;
  }
  const char* kind = PyUnicode_AsUTF8(kind_object);
  if (!kind) return nullptr;
  if (dmax < 1 || eps < 0.0) {
    PyErr_SetString(PyExc_ValueError, "invalid native MPS truncation options");
    return nullptr;
  }
  ArrayView left;
  ArrayView right;
  if (!get_tensor(left_object, &left, "left tensor") ||
      !get_tensor(right_object, &right, "right tensor")) {
    release_array_view(&left);
    release_array_view(&right);
    return nullptr;
  }
  if (left.dr != right.dl) {
    release_array_view(&left);
    release_array_view(&right);
    PyErr_SetString(PyExc_ValueError, "native MPS tensors have incompatible bonds");
    return nullptr;
  }
  std::vector<Complex> payload;
  bool payload_ok = false;
  if (std::strcmp(kind, "generic") == 0) payload_ok = read_gate(payload_object, &payload);
  else if (std::strcmp(kind, "diagonal") == 0 || std::strcmp(kind, "zz") == 0)
    payload_ok = read_values(payload_object, 4, &payload);
  else payload_ok = true;
  if (!payload_ok) {
    release_array_view(&left);
    release_array_view(&right);
    return nullptr;
  }
  std::vector<Complex> tensor;
  merge_two_site(left, right, &tensor);
  apply_payload(&tensor, left.dl, right.dr, payload, kind);
  SplitResult split;
  const bool ok = split_tensor(
      tensor, left.dl, right.dr, dmax, static_cast<float>(eps),
      renormalize != 0, &split);
  const int dl = left.dl;
  const int dr = right.dr;
  release_array_view(&left);
  release_array_view(&right);
  if (!ok) return nullptr;
  PyObject* left_array = make_array(split.left, {dl, 2, split.rank_kept});
  PyObject* right_array = make_array(split.right, {split.rank_kept, 2, dr});
  if (!left_array || !right_array) {
    Py_XDECREF(left_array);
    Py_XDECREF(right_array);
    return nullptr;
  }
  PyObject* metadata = Py_BuildValue(
      "{s:i,s:i,s:f,s:f,s:f,s:s,s:[]}", "rank_before", split.rank_before,
      "rank_kept", split.rank_kept, "local_discarded_weight",
      split.discarded_weight, "relative_discarded_weight",
      split.total_weight > 0.0f ? split.discarded_weight / split.total_weight : 0.0f,
      "pre_normalization_norm", std::sqrt(split.kept_weight),
      "svd_driver", "accelerate_cgesdd", "svd_failed_attempts");
  if (!metadata) {
    Py_DECREF(left_array);
    Py_DECREF(right_array);
    return nullptr;
  }
  PyObject* result = PyTuple_New(3);
  if (!result) {
    Py_DECREF(left_array);
    Py_DECREF(right_array);
    Py_DECREF(metadata);
    return nullptr;
  }
  PyTuple_SET_ITEM(result, 0, left_array);
  PyTuple_SET_ITEM(result, 1, right_array);
  PyTuple_SET_ITEM(result, 2, metadata);
  return result;
}

std::vector<int> candidate(int first, int second, bool move_first) {
  std::vector<int> swaps;
  if (first < second) {
    if (move_first) {
      for (int site = first; site < second - 1; ++site) swaps.push_back(site);
    } else {
      for (int site = second - 1; site > first; --site) swaps.push_back(site);
    }
  } else if (move_first) {
    for (int site = first - 1; site > second; --site) swaps.push_back(site);
  } else {
    for (int site = second; site < first - 1; ++site) swaps.push_back(site);
  }
  return swaps;
}

std::vector<int> simulate(const std::vector<int>& order,
                          const std::vector<int>& swaps) {
  std::vector<int> result = order;
  for (int site : swaps) std::swap(result[site], result[site + 1]);
  return result;
}

double cost(const std::vector<int>& order, const std::vector<Pair>& pairs) {
  std::vector<int> positions(order.size(), -1);
  for (int site = 0; site < static_cast<int>(order.size()); ++site) {
    positions[order[site]] = site;
  }
  double total = 0.0;
  for (size_t index = 0; index < pairs.size(); ++index) {
    const auto [first, second] = pairs[index];
    const int distance = std::abs(positions[first] - positions[second]);
    total += static_cast<double>(std::max(0, distance - 1)) /
             static_cast<double>(index + 1);
  }
  return total;
}

PyObject* make_int_list(const std::vector<int>& values) {
  PyObject* result = PyList_New(static_cast<Py_ssize_t>(values.size()));
  if (!result) return nullptr;
  for (size_t index = 0; index < values.size(); ++index) {
    PyObject* value = PyLong_FromLong(values[index]);
    if (!value) {
      Py_DECREF(result);
      return nullptr;
    }
    PyList_SET_ITEM(result, static_cast<Py_ssize_t>(index), value);
  }
  return result;
}

PyObject* plan_routing(PyObject*, PyObject* args) {
  int n_qubits = 0;
  int lookahead = 0;
  PyObject* pairs_object = nullptr;
  if (!PyArg_ParseTuple(args, "iOi", &n_qubits, &pairs_object, &lookahead)) {
    return nullptr;
  }
  if (n_qubits < 1 || lookahead < 0) {
    PyErr_SetString(PyExc_ValueError, "n_qubits must be positive and lookahead non-negative");
    return nullptr;
  }

  PyObject* pairs_fast = PySequence_Fast(pairs_object, "pairs must be a sequence");
  if (!pairs_fast) return nullptr;
  std::vector<Pair> pairs;
  const Py_ssize_t pair_count = PySequence_Fast_GET_SIZE(pairs_fast);
  pairs.reserve(static_cast<size_t>(pair_count));
  for (Py_ssize_t index = 0; index < pair_count; ++index) {
    PyObject* pair_object = PySequence_Fast_GET_ITEM(pairs_fast, index);
    PyObject* pair_fast = PySequence_Fast(pair_object, "each pair must be a sequence");
    if (!pair_fast) {
      Py_DECREF(pairs_fast);
      return nullptr;
    }
    if (PySequence_Fast_GET_SIZE(pair_fast) != 2) {
      Py_DECREF(pair_fast);
      Py_DECREF(pairs_fast);
      PyErr_SetString(PyExc_ValueError, "each routing pair must have two wires");
      return nullptr;
    }
    const long first = PyLong_AsLong(PySequence_Fast_GET_ITEM(pair_fast, 0));
    const long second = PyLong_AsLong(PySequence_Fast_GET_ITEM(pair_fast, 1));
    Py_DECREF(pair_fast);
    if (PyErr_Occurred()) {
      Py_DECREF(pairs_fast);
      return nullptr;
    }
    if (first < 0 || second < 0 || first >= n_qubits || second >= n_qubits || first == second) {
      Py_DECREF(pairs_fast);
      PyErr_SetString(PyExc_ValueError, "routing wires are outside the MPS");
      return nullptr;
    }
    pairs.emplace_back(static_cast<int>(first), static_cast<int>(second));
  }
  Py_DECREF(pairs_fast);

  std::vector<int> order(static_cast<size_t>(n_qubits));
  for (int index = 0; index < n_qubits; ++index) order[index] = index;
  PyObject* plan = PyList_New(0);
  if (!plan) return nullptr;
  int swap_count = 0;
  try {
    for (size_t index = 0; index < pairs.size(); ++index) {
      const auto [first, second] = pairs[index];
      std::vector<int> positions(static_cast<size_t>(n_qubits), -1);
      for (int site = 0; site < n_qubits; ++site) positions[order[site]] = site;
      const auto first_swaps = candidate(positions[first], positions[second], true);
      const auto second_swaps = candidate(positions[first], positions[second], false);
      const auto first_order = simulate(order, first_swaps);
      const auto second_order = simulate(order, second_swaps);
      std::vector<Pair> future;
      const size_t stop = std::min(pairs.size(), index + 1 + static_cast<size_t>(lookahead));
      for (size_t future_index = index + 1; future_index < stop; ++future_index) {
        future.push_back(pairs[future_index]);
      }
      const bool choose_first = cost(first_order, future) <= cost(second_order, future);
      const auto& selected = choose_first ? first_swaps : second_swaps;
      order = choose_first ? first_order : second_order;
      PyObject* swap_list = make_int_list(selected);
      if (!swap_list || PyList_Append(plan, swap_list) < 0) {
        Py_XDECREF(swap_list);
        Py_DECREF(plan);
        return nullptr;
      }
      Py_DECREF(swap_list);
      swap_count += static_cast<int>(selected.size());
    }
  } catch (const std::exception& error) {
    Py_DECREF(plan);
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
  PyObject* order_list = make_int_list(order);
  if (!order_list) {
    Py_DECREF(plan);
    return nullptr;
  }
  PyObject* result = PyTuple_New(3);
  if (!result) {
    Py_DECREF(plan);
    Py_DECREF(order_list);
    return nullptr;
  }
  PyTuple_SET_ITEM(result, 0, plan);
  PyTuple_SET_ITEM(result, 1, order_list);
  PyTuple_SET_ITEM(result, 2, PyLong_FromLong(swap_count));
  return result;
}

PyMethodDef methods[] = {
    {"plan_routing", plan_routing, METH_VARARGS, "Plan MPS route swaps."},
    {"two_site_update", native_two_site_update, METH_VARARGS,
     "Merge, update, split, and truncate two MPS sites in native CPU code."},
    {"qr_move_right", native_qr_move_right, METH_VARARGS,
     "Move the MPS canonical center right with Accelerate QR."},
    {"qr_move_left", native_qr_move_left, METH_VARARGS,
     "Move the MPS canonical center left with Accelerate QR."},
    {"expectation_product", native_expectation_product, METH_VARARGS,
     "Evaluate a mapping-aware product observable in native CPU code."},
    {"probabilities", native_probabilities, METH_VARARGS,
     "Evaluate mapping-aware MPS marginal probabilities in native CPU code."},
    {"sample", native_sample, METH_VARARGS,
     "Draw mapping-aware MPS samples in native CPU code."},
    {"statevector", native_statevector, METH_VARARGS,
     "Materialize a logical-order statevector in native CPU code."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_mps_native", "Optional native MPS helpers.", -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__mps_native() {
  if (_import_array() < 0) return nullptr;
  return PyModule_Create(&module);
}
