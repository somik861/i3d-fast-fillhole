/*
 * i3dlib - image manipulation library
 *
 * Copyright (C) 2000-2006   Centre for Biomedical Image Analysis (CBIA)
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the Free
 * Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * tp_nha_domain initializer.h
 *
 * Initializer of the topology-preserving version of the Nilsson-Heyden algorithm with respect to given domain
 *
 * Martin Ma�ka (xmaska@fi.muni.cz) 2006 - 2011
 */

#ifndef FLS_TP_NHA_DOMAIN_INITIALIZER_H
#define FLS_TP_NHA_DOMAIN_INITIALIZER_H

#include "nha_domain_initializer.h"

namespace i3d
{
	namespace fls
	{
		/////////////////////////////////////////////////////////////////
		//                                                             //
		//                Class TP_NHA_DomainInitializer               //
		//                                                             //
		/////////////////////////////////////////////////////////////////

		/** Initializer of the topology-preserving version of the Nilsson-Heyden algorithm for 2D grids with respect to given domain. */
		class TP_NHA_DomainInitializer : public NHA_DomainInitializer
		{
			public:
				/** Constructor. */
				TP_NHA_DomainInitializer(Grid<int> &grid, const Type::BinVoxel *mask, const Type::BinVoxel *domain) : NHA_DomainInitializer(grid, mask, domain) {}; 
				/** Initialize the implicit function and find grid points belonging to the interface. */
				virtual void Initialize(std::vector<Type::Size> &initial_interface);
		};

		/////////////////////////////////////////////////////////////////
		//                                                             //
		//               Class TP_NHA_DomainInitializer3D              //
		//                                                             //
		/////////////////////////////////////////////////////////////////

		/** Initializer of the topology-preserving version of the Nilsson-Heyden algorithm for 3D grids with respect to given domain. */
		class TP_NHA_DomainInitializer3D : public TP_NHA_DomainInitializer
		{
			public:
				/** Constructor. */
				TP_NHA_DomainInitializer3D(Grid<int> &grid, const Type::BinVoxel *mask, const Type::BinVoxel *domain) : TP_NHA_DomainInitializer(grid, mask, domain) {};
				/** Initialize the implicit function and find grid points belonging to the interface. */
				virtual void Initialize(std::vector<Type::Size> &initial_interface);
		};

	} // namespace fls

} // namespace i3d

#endif // FLS_TP_NHA_DOMAIN_INITIALIZER_H

